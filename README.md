# Multigrid Monte Carlo
C++ implementation of multigrid Monte Carlo algorithm

## Dependencies
The code requires the [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page) for linear algebra as well as [libconfig](https://hyperrealm.github.io/libconfig/) for parsing configuration files. To install libconfig, clone the [libconfig repository](https://github.com/hyperrealm/libconfig) and build/install it with CMake (this will then allow the Multigrid MC code to find libconfig with `find_package()` in the CMake configure step).

## Building the code
To compile, create a new directory called `build`. Change to this directory and run

```
cmake ..
```

to configure, followed by

```
make
```

to build the code.

## Running the code
The executable is `driver` in the `bin` subdirectory. To run the code, use

```
./bin/driver CONFIG_FILE
```

where `CONFIG_FILE` is the name of the file that contains the runtime configuration; an example can be found in [parameters_template.cfg](parameters_template.cfg).