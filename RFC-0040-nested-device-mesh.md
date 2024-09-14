# NestedDeviceMesh

**Authors:**
* @Jackmin801
* @samsja 
* @JohannesHa

## **Summary**
The current abstraction for managing process groups in torch distributed is DeviceMesh, which provides a useful model for defining N-dimensional parallelisms.
However, the current implementation has some limitations that make it unsuitable for heterogenous training workloads.
Namely that the DeviceMesh needs to be cuboid, assuming homogeneity among processes.
This restriction limits its applicability for heterogeneous training workloads, where different devices might have varying workloads that do not map neatly onto a cuboid mesh.

To address this limitation, we propose the introduction of a NestedDeviceMesh abstraction.
This new abstraction allows for more flexibility by allowing nesting of DeviceMesh instances, an element of a DeviceMesh can be a DeviceMesh, each potentially with different shapes and sizes.
This would allow users to model more complex parallelism strategies, such as those required by heterogeneous workloads.

## **Motivation**
This proposal aims to address the growing need for efficient distributed training in [heterogeneous environments](https://arxiv.org/abs/2301.11913), particularly as deep learning models continue to scale.
Training workloads can involve a mix of hardware types, ranging from smaller clusters with the newest GPUs to big clusters containing older GPUs.

Large-scale distributed training in heterogeneous environments faces two key challenges: [communication overhead and the straggler effect](https://www.semianalysis.com/p/multi-datacenter-training-openais).
In multi-region setups with diverse GPU configurations, synchronization delays can significantly increase, especially when slower or less powerful GPUs hold up the entire training process.

The NestedDeviceMesh abstraction helps mitigate these issues by enabling a hierarchical structure where different submeshes handle synchronization independently.
By assigning smaller workloads to slower GPUs and larger workloads to more powerful ones, the system reduces the impact of stragglers, ensuring that weaker devices don't slow down the entire training process.
This design allows for efficient scaling across heterogeneous hardware, minimizing global synchronization steps and improving overall training efficiency.

By implementing NestedDeviceMesh, we aim to support:
- Optimization algorithms that reduce overall communication latency by limiting synchronization to smaller, hierarchical submeshes [[1](https://arxiv.org/abs/2407.07852), [2](https://arxiv.org/abs/2311.08105), [3](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf)].
- Better support heterogeneous hardware, allowing for flexibility in scaling across different architectures, GPU counts, and regions.

This design will be particularly beneficial for multi-data center setups where inter-region latency makes frequent synchronization prohibitively expensive.
By enabling more independent progress within submeshes and reducing the number of global synchronization steps, NestedDeviceMesh enables more efficient distributed training at scale.


## **Proposed Implementation**
The NestedDeviceMesh will extend the existing DeviceMesh class, adding the capability to define a hierarchy of meshes.
Each level of the hierarchy will represent a different mesh, with the possibility of defining different dimensions and sizes for each mesh.

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

## **Drawbacks**
- **Impact on UX**: The introduction of nested meshes could complicate the user experience, especially for those who have existing code bases with the old API. However, they can opt-out by default by not using the new DeviceMesh class.
- **Integration Challenges**: Ensuring compatibility with all existing and future distributed training features in PyTorch will require coordination with the torch distributed team.

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
- **API Consistency**: How can we ensure that the NestedDeviceMesh API is consistent with other PyTorch abstractions?
- **Supported Parallelisms**: Which parallelisms should be supported on the outer mesh at time of release?
- **Backends**: Which backends should we support?

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
