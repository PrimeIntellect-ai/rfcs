# (Dynamic)NestedDeviceMesh

**Authors:**
* @Jackmin801
* @samsja 
* @JohannesHa

## **Summary**
The current abstraction for managing process groups in torch distributed is DeviceMesh, which provides a useful model for defining N-dimensional parallelisms.
However, the current implementation has some limitations that make it unsuitable for dynamic heterogenous training workloads.
The first limitation is that the DeviceMesh needs to be cuboid, assuming homogeneity among processes.
This restriction limits its applicability for heterogeneous training workloads, where different devices might have varying workloads that do not map neatly onto a cuboid mesh.
The second limitation is that the world size needs to be known at start time. This results in an inability to create dynamic training runs where groups of processes join and leave the training without restarting all the processes in the training run.

To address the first limitation, we propose the introduction of a NestedDeviceMesh abstraction.
This new abstraction allows for more flexibility by allowing nesting of DeviceMesh instances, an element of in a DeviceMesh can be a DeviceMesh, each potentially with different shapes and sizes.
This would allow users to model more complex parallelism strategies, such as those required by heterogeneous workloads.

To address the second limitation, we propose the process group creation API be rewritten to support dynamic world size by periodically polling the c10d store for the current world size and recreating the nccl communicators in the case of a change.

## **Motivation**
This proposal is motivated by the need to support dynamic mixed hardware in OpenDiLoCo.

The goal of OpenDiLoCo is to reduce the network requirement of deep learning optimization by utilising an inner and outer optimization.
Each participating worker will optimize their respective copies of the model with the inner optimization without communicating with other workers.
The outer optimisation step which requires communication between all workers is only triggered every few hundred steps -- reducing the amount of all worker communications.

OpenDiLoCo plans to support on and off ramping of workers which requires the outer world size to be dynamic.
It also plans to support the participation of heterogenous workers (e.g. a worker can have 8 H100 or 16 A100) in a single training run which requires the NestedDeviceMesh abstraction.

## **Proposed Implementation**
The NestedDeviceMesh will extend the existing DeviceMesh class, adding the capability to define a hierarchy of meshes.
Each level of the hierarchy will represent a different mesh, with the possibility of defining different dimensions and sizes for each mesh.

The implementation of NestedDeviceMesh will also support dynamic world sizes.
Allowing processes to join and leave the mesh before the collective communication is called.

### Key Components:
1. **NestedDeviceMesh:**
   - A NestedDeviceMesh can contain multiple DeviceMesh objects, each with its configuration.

### Code Example

DeviceMesh usage
```python
from torch.distributed.device_mesh import init_device_mesh
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))

# Users can access the underlying process group thru `get_group` API.
replicate_group = mesh_2d.get_group(mesh_dim="replicate")
shard_group = mesh_2d.get_group(mesh_dim="shard")
```

NestedDeviceMesh usage
```python
from torch.distributed.device_mesh import init_device_mesh

# Initialize the main outer DeviceMesh
outer_mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("outer"), rdzv_endpoint="10.0.0.1:29400")
# Initialize the submesh from the main mesh
# The submesh is a smaller cuboid inside the larger mesh
outer_mesh.init_submesh((torch.cuda.device_count(),), mesh_dim_names=("replicate"), rdzv_endpoint="127.0.0.1:29401")

# Access the group from the main mesh
main_group = main_mesh.get_group(mesh_dim="outer")

# Access the submesh and its group through the main mesh
submesh_group = main_mesh.get_submesh().get_group(mesh_dim="shard")
```

### Handling Edge Cases
- Error handling for invalid mesh configurations

## **Metrics**
- **Performance parity with DeviceMesh for static training run**: The dynamic NestedDeviceMesh should have the same performance characteristics as the original DeviceMesh in the setup where the dynamic property is not utilised.
- **Performance improvement compared to DeviceMesh for dynamic training runs**: The dynamic NestedDeviceMesh should have better performance characteristic compared to the original DeviceMesh in the setup where the dynamic property is utilised. It should be cheaper to recreate process groups than it is to restart all the workers with elastic agent.
- **Performance improvement compared to OpenDiLoCo hivemind implementation**: The new method for orchestrating OpenDiLoCo training should have better performance than the original one which utilises hivemind.
- **Must allow on and off ramping of processes**: It should be possible for processes to leave the outer DeviceMesh without ruining the training run. It should be possible to join the outer DeviceMesh without ruining the training run.

## **Drawbacks**
- **Impact on UX**: The introduction of nested meshes could complicate the user experience, especially for those who have existing code bases with the old API. However, they can opt-out by default by not using the new DeviceMesh class.
- **Maintenance Overhead**: The dynamic feature might require a change to the way torch manages process groups that is not backwards compatible, requiring two different implementation paths which increases maintenance overhead.
- **Integration Challenges**: Ensuring compatibility with all existing and future distributed training features in PyTorch will require coordination with the torch distributed team.
- **Implementation Costs**: The feature might take considerable effort to implement. However, we think the effort is worth the benefits. Namely that it becomes easy for process groups to be dynamically recreated and hierarchical optimizers (like OpenDiLoCo) are supported.

## **Alternatives**
- **Multi world abstraction instead of nesting**: Another alternative is to allow the creation of separate heterogeneous process groups directly, without requiring a mesh abstraction. However, this could complicate the mental model in the future if we plan to support communication between NestedDeviceMeshes (say to support PP in the outer mesh which would currently only support DP).

## **Prior Art**

- [pymultiworld](https://github.com/cisco-open/pymultiworld) is a library that implemented a patch on torch that allowed process to be part of multiple worlds.
- [OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo) implemented dual-world by having the outer world use the [hivemind](https://github.com/learning-at-home/hivemind) backend while the inner world used the torch nccl backend.

## **How we teach this**
Teaching NestedDeviceMesh would involve:

- **Documentation Updates**: Detailed documentation with examples illustrating common use cases, such as OpenDiLoCo training.
- **Tutorials and Examples**: Providing Jupyter notebooks and scripts that demonstrate how to set up and use NestedDeviceMesh in real-world scenarios.
- **Terminology**: Clear definitions of new terms introduced by this feature, such as "NestedDeviceMesh", "Hierarchical Parallelism", etc.
- **Community Engagement**: Engaging with the PyTorch community through forums, blog posts, and webinars to explain the benefits and usage of NestedDeviceMesh.

Accepting this proposal would require updates to the PyTorch documentation to include the new NestedDeviceMesh class and its associated methods. Existing guides on distributed training could also be changed to use the new NestedDeviceMesh as a default if the UX is simpler or equivalent to the original DeviceMesh

## **Unresolved questions**
- **Characteristics of dynamic process groups**: Is dynamically recreating process groups performant and fault-tolerant?
- **API Consistency**: How can we ensure that the NestedDeviceMesh API is consistent with other PyTorch abstractions?
- **Supported Parallelisms**: Which parallelisms should be supported on the outer mesh at time of release?
- **Integration with Other Backends**: Will this work for none nccl backends?

## **Resolution**
TBD

### Level of Support
TBD

#### Additional Context
TBD

### Next Steps
TBD

#### Tracking issue
TBD
