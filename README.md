JPEG-LS HDF5 Filter
===================
The JPEG-LS HDF5 filter allows the multi-threaded compression of HDF5 datasets using the JPEG-LS codec.

Dependencies
------------

On Ubuntu 18.04 or above, install the following packages:
```bash
sudo apt install build-essential ninja meson hdf5-tools libhdf5-dev
```

The filter depends on the CharLS implementation of JPEG-LS,
https://github.com/team-charls/charls
It is automatically downloaded by the Meson build system during the configuration process.

Building
--------
1. First, configure the build system:

    ```bash
    meson --buildtype=release -Db_lto=true build/release
    ```

1. Build the hdf5 filter plugin:

    ```bash
    ninja -C build/release
    ```

1. (Optional) Run tests:

    ```bash
    ninja -C build/release test
    ```

(TBD) Installation
-------------------

Execute the following to install the plugin:

```bash
ninja -C build/release install
```
when prompted, enter admin password to continue.