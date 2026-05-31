//Numpy array shape [3, 1, 1]
//Min -0.500000000000
//Max 1.375000000000
//Number of zeros 0

#ifndef W8_H_
#define W8_H_

#ifndef __SYNTHESIS__
weight8_t w8[3];
#else
weight8_t w8[3] = {-0.50000, 1.37500, 0.68750};
#endif

#endif
