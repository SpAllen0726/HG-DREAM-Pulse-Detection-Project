//Numpy array shape [4]
//Min -0.031250000000
//Max 0.187500000000
//Number of zeros 0

#ifndef B8_H_
#define B8_H_

#ifndef __SYNTHESIS__
bias8_t b8[4];
#else
bias8_t b8[4] = {0.18750, -0.03125, 0.12500, 0.12500};
#endif

#endif
