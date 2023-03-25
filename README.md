# Multigrid Monte Carlo
C++ implementation of multigrid Monte Carlo algorithm

## Dependencies
The code requires the [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page).

## Building the code
To compile, change to the `src` directory and run

```
cmake ..
```

to configure, followed by

```
make
```

to build the code.

## Running the code
The executable is `driver` in the `src` directory. To run the code, use

```
./driver NX NY
```

where `NX`, `NY` specifies the lattice size.