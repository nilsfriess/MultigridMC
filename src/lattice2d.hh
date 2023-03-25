#ifndef LATTICE2D_HH
#define LATTICE2D_HH LATTICE2D_HH

class Lattice2d
{
public:
  Lattice2d(const unsigned int nx_, const unsigned int ny_) : nx(nx_), ny(ny_), M(nx_ * ny_) {}

  const unsigned int nx;
  const unsigned int ny;
  const unsigned int M;
};

#endif // LATTICE2D_HH