//Numpy array shape [16]
//Min -0.031250000000
//Max 0.062500000000
//Number of zeros 6

#ifndef B8_H_
#define B8_H_

#ifndef __SYNTHESIS__
bias8_t b8[16];
#else
bias8_t b8[16] = {0.03125, 0.03125, 0.03125, -0.03125, 0.03125, 0.03125, 0.06250, 0.00000, -0.03125, -0.03125, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.03125};
#endif

#endif
