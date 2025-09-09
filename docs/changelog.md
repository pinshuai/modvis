# Changelog

## v0.0.9 - September 7, 2025

This release includes new visualization capability for mixed element meshes (thanks to Saubhagya Rathore@ORNL!). Note this is still work in progress and package will be updated to include more plotting functions.

### What's Changed
* Added visualization for mixed element meshes by @pinshuai in https://github.com/pinshuai/modvis/pull/4
* Updated nwis function
* updated license from MIT to GNU GPL
* fixed a few bugs in loading meshes

## v0.0.8 - July 16, 2024

In this release, edits to two functions are made: plot_column_data() and plot_column_head(). A new argument was added to allow inferring column ids from cell ids. This is convenient when column ids cannot be easily obtained.

## v0.0.7 - July 27, 2023
- Updated VisFile class to work with ats version >1.5
- Merged ats_xdmf.py from ats repo

## v0.0.3 - May 18, 2022

**Improvements**

- Clean up the requirements.txt to make all dependencies installable through `pip`
- All examples now works with the `import modvis` statement

## v0.0.2 - May 18, 2022

Clean up the load visfile function to make the workflow much cleaner.

## v0.0.1 - May 2, 2022

First release of the package.
