#include <Kokkos_Core.hpp>
struct functor {
  functor( void * ptr_d_ ) : ptr_d((unsigned long)ptr_d_), mult(2.0) {}

  double mult;   
  unsigned long ptr_d=0xdeadbeef;


  inline
  void operator()(const int i,  cl::sycl::stream out) const {
    double *dptr = (double *)ptr_d;

    if ( dptr == 0x0 ) {
      if ( i  == 2 ) { 
	out << "Id = " << i << " BARF!!!" << cl::sycl::endl;
      }
    }
    else {
      dptr[i] = mult*(double)i+1.5;
    }
  }
};

void copyAndPrint(void *ptr_d, cl::sycl::queue* q, int N);

#include "KokkosProxies.hpp"

int main(int argc, char *argv[])
{
	Kokkos::initialize(argc,argv);

	cl::sycl::queue* q = Kokkos::Experimental::SYCL().impl_internal_space_instance()->m_queue;
	auto context = q->get_context();
	auto  device = q->get_device();

	const int N = 15;

	void* ptr_d=cl::sycl::malloc_device(N*sizeof(double),device,context);
	std::cout << " q ptr is : " << (unsigned long)q << std::endl;
	std::cout << " ptr_d is : " << (unsigned long)ptr_d << std::endl;
	std::cout << " Calling q.submit() " << std::endl;

	functor f(ptr_d);
	f.mult=4.0;
	Foo::Bar::my_parallel_for_2(N,f);
	copyAndPrint(ptr_d,q,N);

	std::cout << "Kokkos::parallel_for " << std::endl;
	f.mult = 6.0;
	Kokkos::parallel_for(N,f);
	Kokkos::fence();
	copyAndPrint(ptr_d,q,N);

	cl::sycl::free((void *)ptr_d,context);
	Kokkos::finalize();
}

void copyAndPrint(void *ptr_d, cl::sycl::queue* q, int N) {


	cl::sycl::buffer<double,1> hbuf(N);
	q->submit([&](cl::sycl::handler& cgh) {
	     auto access = hbuf.get_access<cl::sycl::access::mode::write>(cgh);
	     cgh.single_task<class copy>([=]{
	       for(int i=0; i<N; i++) {
                 access[i] = ((double *)ptr_d)[i];
	       }
	     });
	 });

	auto haccess = hbuf.get_access<cl::sycl::access::mode::read>();
	for(int i=0; i < N; ++i) {
	  printf("%d %lf\n", i, haccess[i]);
	}

}

