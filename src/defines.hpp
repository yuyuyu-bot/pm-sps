#ifndef __PMS_DEFINES__
#define __PMS_DEFINES__

#define _CLOCK 0

#ifdef _WIN32
#define popcnt32 __popcnt
#define popcnt64 __popcnt64
#else
#define popcnt32 __builtin_popcount
#define popcnt64 __builtin_popcountll
#endif

#endif