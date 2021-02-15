# Running `fit2d` in the command line

### Why?
- Easier to share code and reproduce experiments
- More convenient if you have to run the code on remote environments (i.e. not your computer)

### Quick start
1. Write a configuration file that contains all the information that you provided in the top cells of the notebook. For an example, see the file `example_config.yml` that is in this directory. When creating a new file, you may want to start by copying this example to use as a template.
2. In the command line, run `python -m fit2d.run <path to config file>`
3. The output pickled samplers will be saved to `<save_dir>/*.pkl` where `<save_dir>` is an optional path you can provide in the configuration. If not specified, outputs will be written to the directory where you ran the command.

### Best practices
- If you run a new experiment, **make a new config file for it**. Don't just modify and overwrite an existing configuration, as it will be hard to track what outputs were produced with what settings.

- If you haven't already, I suggest you come up with a shared convention for naming the galaxy fits files, e.g. `<galaxy>_mom1.fits, <galaxy>_mom2.fits>` and then always save them in a directory with the galaxy's name.

- Create a master `experiments` directory. Each time you run a new experiment, create a new subdirectory within `experiments` with a descriptive name, e.g. `experiments/UGC1234_const_err_fill_nan_pixels`. Create the new configuration for the experiment within its directory, and make sure to `cd` into this directory before running the command.

