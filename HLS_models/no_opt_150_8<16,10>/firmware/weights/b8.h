//Numpy array shape [8]
//Min -0.062500000000
//Max 0.125000000000
//Number of zeros 3

#ifndef B8_H_
#define B8_H_

#ifndef __SYNTHESIS__
bias8_t b8[8];
#else
bias8_t b8[8] = {0.00000, 0.03125, -0.06250, 0.00000, 0.12500, -0.03125, 0.00000, 0.03125};
#endif

#endif
