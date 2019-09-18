#include <Kokkos_Core.hpp>


struct functor {
 
  double mult;   
  double* ptr_d;

  void operator()(const int i,  cl::sycl::stream out) const {

    if ( ptr_d == 0x0 ) {
      if ( i  == 2 ) { 
	out << "Id = " << i << " BARF!!!" << cl::sycl::endl;
      }
    }
    else {
      ptr_d[i] = mult*(double)i+1.5;
    }
  }

  friend std::ostream& operator<<(std::ostream& os, functor const& that)
  { return os << "my addr=" << &that << " mult=" << that.mult << " ptr_d=" << that.ptr_d; }

  friend cl::sycl::stream operator<<(cl::sycl::stream os, functor const& that)
  { return os << "my addr=" << &that << " mult=" << that.mult << " ptr_d=" << that.ptr_d; }

};

void copyAndPrint(void *ptr_d, cl::sycl::queue* q, int N);
void zero(double *ptr_d, cl::sycl::queue *q, int N);

#include "KokkosProxies.hpp"

int main(int argc, char *argv[])
{
	Kokkos::initialize(argc,argv);

	cl::sycl::queue* q = Kokkos::Experimental::SYCL().impl_internal_space_instance()->m_queue;
	auto context = q->get_context();
	auto  device = q->get_device();

	const int N = 15;

	double* ptr_d=(double *)cl::sycl::malloc_device(N*sizeof(double),device,context);
	std::cout << " q ptr is : " << (unsigned long)q << std::endl;
	std::cout << " ptr_d is : " << (unsigned long)ptr_d << std::endl;
	std::cout << " Calling q.submit() " << std::endl;

	functor* f=(functor *)cl::sycl::malloc_shared(sizeof(functor),device,context);
	f->mult=4.0;
	f->ptr_d = ptr_d;
	Foo::Bar::my_parallel_for_2(N,*f);
	copyAndPrint(ptr_d,q,N);

	std::cout << "Kokkos::parallel_for " << std::endl;
	f->mult = 6.0;
	f->ptr_d = ptr_d;

	Kokkos::parallel_for(N,*f);
	Kokkos::fence();
	copyAndPrint(ptr_d,q,N);

#ifndef NO_LAMBDA
	std::cout << "Kokkos::oarallel for with lambda " << std::endl;
	Kokkos::parallel_for(N,[=](const int i,  cl::sycl::stream out) {
	   double mult=9;

           if ( ptr_d == 0x0 ) {
             if ( i  == 2 ) {
               out << "Id = " << i << " BARF!!!" << cl::sycl::endl;
             }
           }
           else {
             ptr_d[i] = mult*(double)i+1.5;
           }
        });
	Kokkos::fence();
	copyAndPrint(ptr_d,q,N);
#endif
	cl::sycl::free((void *)ptr_d,context);
	cl::sycl::free((void *)f,context);
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

void zero(double *ptr_d, cl::sycl::queue *q, int N) 
{
        q->submit([&](cl::sycl::handler& cgh) {
             cgh.single_task<class copy>([=]{
               for(int i=0; i<N; i++) {
                ptr_d[i]=0;
               }
             });
         });
}
