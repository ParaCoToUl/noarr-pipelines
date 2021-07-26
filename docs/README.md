# Noarr Documentation

Here's the top-level structure of the entire documentation for Noarr. This is how all the `.md` files will be combined into one PDF document + it wil act as a table of contents with hyperlinks to individual parts:

- Introduction `(what is structures/pipelines, when to use which, note about bindings)`
- Structures showcase `(short teaser)`
- Pipelines showcase `(short teaser, will also include structures)`
- Noarr Structures
    - `(detailed documentation of structures - placed in the other repository)`
    - `Showcase of everything (how to use each feature)`
    - `Advanced stuff (how it works, how to extend, etc...)`
- Noarr Pipelines
    - `(detailed documentation of pipelines - placed in this repository, this folder)`
    - [Core principles](core-principles.md)
    - Compute Node `(event methods, construction, async processing)`
    - [Hub](hub.md) `(envelopes, links, allocation, data transfer, dataflow strategy, direct manipulation)`
    - Cuda Pipelines `(cuda compute node + memory allocation)`
    - Hardware Manager and custom extensions
- Examples
    - `(desription of available examples - placed in the examples folder)`
- Python and R bindings
    - `(desribe how to extend binding templates - placed in the templates folder)`
