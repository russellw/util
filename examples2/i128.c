// argv: 19
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

__uint128_t factorial(__uint128_t n){
	if(n<=1)
		return 1;
		return n*factorial(n-1);
	}


int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: i128 n\n");
    return 1;
  }
  errno = 0;
  uint64_t n = strtoull(argv[1], 0, 10);
  if (errno) {
    perror(argv[1]);
    return 1;
  }

  __uint128_t r=factorial(n);
  assert(r/factorial(n-1)==n);
  printf("%" PRIu64 "\n", (uint64_t)(r/factorial(n-1)));
  return 0;
}
