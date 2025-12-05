# Augmentation Approach and Considerations

## Initial Approach

I actually started this work with manifold expansion, but these augmentation techniques are diffeomorphisms - they fill topological gaps. It's not manifold expansion, it's manifold thickening.

## Domain Knowledge Limitations

I'm not an agricultural engineer, so in an area where domain knowledge is crucial (like leaf spots/lesions), I won't venture into imitation with something like GAN/Stable Diffusion.

### Additionally;
I wrote a small helper script, fix_paths.py, which simply updates the root_dir/image_path entries in the DataFrame to match the current working directory. This is only for making the pipeline more robust and portable across different environments; it doesn’t change the data or labels, just fixes paths so everything runs smoothly.
