//Numpy array shape [16]
//Min -0.093750000000
//Max 0.125000000000
//Number of zeros 3

#ifndef B8_H_
#define B8_H_

#ifndef __SYNTHESIS__
bias8_t b8[16];
#else
bias8_t b8[16] = {0.03125, 0.03125, 0.12500, 0.03125, 0.06250, -0.03125, 0.06250, 0.00000, 0.09375, 0.09375, 0.06250, 0.00000, -0.03125, 0.06250, 0.00000, -0.09375};
#endif

#endif
