# Multigrid Monte Carlo
C++ implementation of multigrid Monte Carlo algorithm

## Dependencies
The code requires the [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page).

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
The executable is `driver` in the `bin` sub directory. To run the code, use

```
./bin/driver NX NY
```

where `NX`, `NY` specifies the lattice size.