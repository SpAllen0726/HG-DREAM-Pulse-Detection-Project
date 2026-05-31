//Numpy array shape [16]
//Min -0.625000000000
//Max 0.437500000000
//Number of zeros 1

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[16];
#else
bias2_t b2[16] = {0.21875, -0.62500, 0.31250, 0.25000, 0.15625, -0.03125, 0.12500, -0.34375, -0.59375, 0.43750, 0.12500, 0.21875, -0.21875, 0.00000, -0.06250, -0.03125};
#endif

#endif
