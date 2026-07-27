# ScANGen Changelog
 * 0.3.0
   - Reads files ending in .h5 or .hdf5
   - Exports Noise dataset but no longer exports the Raw dataset because the noise dataset also supplies the raw data.
 * 0.2.0 - CycleISP and Raw Diffusion
   - Added support for new conversion from RGB to RAW called "Raw Diffusion."
   - Removed support for `generate` to make RAW because it now happens in another repoository.
   - The RAW now starts with no noise, as it should.
   - Added data augmentation for `NoiseCIFARDataset`.
 * 0.1.2 - Bugfix after initial release
 * 0.1.1 - Initial release with CycleISP data
 * 0.1.0 - Internal release
