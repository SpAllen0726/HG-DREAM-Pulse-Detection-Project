//Numpy array shape [16]
//Min -0.312500000000
//Max 0.250000000000
//Number of zeros 1

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[16];
#else
bias2_t b2[16] = {0.25000, 0.15625, -0.09375, 0.09375, 0.21875, 0.06250, 0.12500, 0.00000, 0.12500, 0.03125, 0.06250, 0.21875, 0.03125, -0.31250, -0.12500, 0.03125};
#endif

#endif
