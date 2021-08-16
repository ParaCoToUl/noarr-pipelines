# Snippets of documentation text that should be placed in their proper place


Put this section into `Hub` section and call it `Links` or something like that.
## Nodes and envelope hosting

From the scheduler's point of view it does not matter where the data in the pipeline comes from. It just advances nodes when they can be advanced. But since we typically think of programs as having a data-structure and an algorithm, we devised two types of nodes: hubs and compute nodes. There is currently no need to develop other kinds of data-holding nodes, but if that was to become a need, this section describes the conceptual framework.

Data is exchanged between nodes via links. There is the node that owns the data - the *host*. And the node that acts on the provided data - the *guest* (e.g. the hub hosts envelopes to compute nodes). This terminology governs the structure of a link.

A links starts out with no envelope attached. It is the host's responsibility to acquire an envelope and attach it to the link. Whether the envelope contains data or not depends on the host and the link type can be used to guide the decision. The moment an envelope is attached by the host, the link transitions to a *fresh* state. Now it is the guest's turn. Guest can look at the envelope and do with it what it sees fit. Once the guest is done, it switches the link to a *processed* state. Now the host can detach the envelope and continue working with it. This is how one node can lend an envelope to another node.

Apart from the link state and the attached envelope, the link also has additional information present that may or may not be used by both interacting sides. The link has a type and it has a *committed* flag. There are four types of links: producing, consuming, peeking and modifying. These types describe the kind of interaction the guest wants to make with the data. When producing, the host provides an empty envelope and guest fills it with data. The opposite happens during consumption. During peeking and modification the host provides some existing data to the guest, who can either look at or modify the data. The *committed* flag is set by the guest during production, consumption or modification to signal that the action was in fact performed. This lets the guest to e.g. not consume a chunk of data if it does not like the chunk.

A link also has an *autocommit* flag that simply states that the action will be committed automatically when the envelope lending finishes. If this flag is set to `false`, the guest node has to manually call a `commit()` method before finishing the envelope lending.

The link also has an associated device, that specifies on which device should the hosted envelope exist.

To learn more about the envelope hosting interplay, read the `Link.hpp` file and the `ComputeNode.hpp` file.
