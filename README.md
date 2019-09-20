# Lambda Capture / USM Pointer Copy Bug

## Introduction

This bugreport shows a lambda capture / USM pointer copy bug we found with the Public Intel SYCL compiler and USM.
The bug is as follows: sometimes a device pointer allocated by `malloc_device` is not 
copied correctly into device code (in a parallel for).

In this example this happens when the pointer is accessed in a *functor* struct
or in a *lambda*, within another lambda, specifically in the prototype Kokkos SYCL Back end dispatch launcher.

We have found a workaround, which was to capture the pointer not as a pointer type but 
as an appropriate integer type (which is just copied), e.g. `uintptr_t`. However, even in this
case the functor needed to be made into something local for capture into the parallel for by value.

## Compilers and drivers

I used the following version of the GitHub compiler:

```
 clang version 9.0.0 (https://github.com/intel/llvm.git 51a6204c09f8c8868cb9675c1a7f4c1386eb3c65)
 Target: x86_64-unknown-linux-gnu
 Thread model: posix
 InstalledDir: /dist/sfw/ubuntu/intel-llvm9-sycl-2019-09-17/bin
```

and I use the Intel HD Compute driver (NEO) version `19.36.14103`


## Getting the example

This example contains the prototype Kokkos back end as a sub-module. Correspondingly
it should be cloned recursively:

```
 git clone --recursive https://github.com/bjoo/parallel_for.git
```

## Manifesting the bug

First we show that unless we make a local copy of the passed in functor between `q.submit` and
the `cgh.parallel_for` the value of a pointer can get lost.  To show this edit the `Makefile`
and set the `TEST_OPTIONS` in line 1 to

```
TEST_OPTIONS=-DNO_LAMBDA
```

edit the `main.cpp` and make the test use a pointer type by commenting in the line:
```
    typedef double* my_ptr_t;
```
and commenting out the other definition like so
``` 
   // typedef uintptr_t my_ptr_t;
```

then build and run with 
```
make clean
make -j 
./bytes_and_flops.host
```

Sometimes the code will pass e.g.  below:

```
Calling Initialize internal with arguments.device_id =-1
Use GPU <=-1 branch 
In impl_iniitialize with config.sycl_device_id = 0
Calling inititalize with sycl_device_id = 0
The system contains 3 devices
Device 0 : Intel(R) Gen9 HD Graphics NEO Driver Version: 19.36.14103 Is accelerator: YES
Device 1 : Intel(R) Core(TM) i7-6770HQ CPU @ 2.60GHz Driver Version: 18.1.0.0920 Is accelerator: NO
Device 2 : SYCL host device Driver Version: 1.2 Is accelerator: NO
Selecting device: 0
Device 0 : Intel(R) Gen9 HD Graphics NEO Driver Version: 19.36.14103 Is accelerator: YES
 q ptr is : 39482240
 ptr_d is : 29384704
 Calling q.submit() 
Calling execute
In sycl_launch_copy
Queue pointer is: 39482240
range=15
driver_in.ptr_d = 29384704
idx = 2 PF ptr_d = 29384704
0 1.500000
1 5.500000
2 9.500000
3 13.500000
4 17.500000
5 21.500000
6 25.500000
7 29.500000
8 33.500000
9 37.500000
10 41.500000
11 45.500000
12 49.500000
13 53.500000
14 57.500000
PASSED
Kokkos::parallel_for 
in execute: ptr_d =  0x1c06000
In sycl_launch
Queue pointer is: 39482240
range=15
driver_in.ptr_d = 29384704
Functor: my addr=0x7ffd419bb230 mult=6 ptr_d=0x1c06000
Before par for: ptr_d=my addr=0x7ffd419bb230 mult=6 ptr_d=0x1c06000
Before par for: local copy = my addr=0x7ffd419bad80 mult=6 ptr_d=0x1c06000

Calling Localfunctor 
0 1.500000
1 7.500000
2 13.500000
3 19.500000
4 25.500000
5 31.500000
6 37.500000
7 43.500000
8 49.500000
9 55.500000
10 61.500000
11 67.500000
12 73.500000
13 79.500000
14 85.500000
PASSED
```

which shows that Kokkos Proxy classes work fine and pass and also the actual Kokkos class also appears to
work. However, please note the line in the output:

```
In par for: Before Kernel call: idx = 2 argument functor : my addr=0x3c92310 mult=0 ptr_d=0x0
In par for: Before Kernel call: idx = 2 localcopy : my addr=0x3c92338 mult=6 ptr_d=0x1c06000
```
which shows that moving from the the `q.submit` to the `cgh.parallel_for` the functor that was passed in had its members *zeroed out* (`mult=0 ptr_d=0`). A local copy made before the `cgh.parallel_for` was entered was captured and seems OK in thread 2 (`idx = 2`) of the parallel for.

However if I run the code a few times pretty soon I will get something like

```
Functor: my addr=0x7ffdb156d650 mult=6 ptr_d=0x24a3000
Before par for: ptr_d=my addr=0x7ffdb156d650 mult=6 ptr_d=0x24a3000
Before par for: local copy = my addr=0x7ffdb156d1a0 mult=6 ptr_d=0x24a3000
In par for: Before Kernel call: idx = 2 argument functor : my addr=0x5f8e310 mult=0 ptr_d=0x0
In par for: Before Kernel call: idx = 2 localcopy : my addr=0x5f8e338 mult=6 ptr_d=0x650065024a3000
Calling Localfunctor 
0 1.500000
1 5.500000
i = 1 diff = 2 > 5.0e-14
2 9.500000
i = 2 diff = 4 > 5.0e-14
3 13.500000
i = 3 diff = 6 > 5.0e-14
4 17.500000
i = 4 diff = 8 > 5.0e-14
5 21.500000
i = 5 diff = 10 > 5.0e-14
6 25.500000
i = 6 diff = 12 > 5.0e-14
7 29.500000
i = 7 diff = 14 > 5.0e-14
8 33.500000
i = 8 diff = 16 > 5.0e-14
9 37.500000
i = 9 diff = 18 > 5.0e-14
10 41.500000
i = 10 diff = 20 > 5.0e-14
11 45.500000
i = 11 diff = 22 > 5.0e-14
12 49.500000
i = 12 diff = 24 > 5.0e-14
13 53.500000
i = 13 diff = 26 > 5.0e-14
14 57.500000
i = 14 diff = 28 > 5.0e-14
MEGABARF!
```
and we can now see that the pointer in the `localcopy` is corrupted for thread with `idx=2`, it has changed from 
`ptr_d=0x24a3000` before the parallel for to `ptr_d=0x650065024a3000`inside it (a bunch of digits seem to have been
masked in/not masked off adding the digits `0x6500650` at the front of the original address. The other threads presumably also got
corrupted pointers hence differences in nearly all the values of `i`. One can get a case where the printed pointer looks
fine but the answers for some threads are still wrong. This is because we only print the pointer for thread with `idx==2`
(originally to reduce clutter on screen)

## The workaround
This workaround was found by @nliber. Edit the `main.cpp` and comment out the typedef aliasing `double*` to `my_ptr_t` and comment in the other definition
like so:
```
// typedef double* my_ptr_t;
typedef uintptr_t my_ptr_t;
```

now make and run again a few times:
```
make clean
make -j 
./bytes_and_flops.host
./bytes_and_flops.host
...
```

the test should now pass all the time. While the functor that was passed in as an argument *is still zeroed out*, 
the localopy now seems to work fine. We hypothesize that this is  because we are claiming that the pointer is now
just an unsigned integer, and any pointer processing doesn't realize it is a pointer and just works. When we need
it we explicitly cast it back to a pointer. We ran a test where we executed the testcase 1000 times and found no
instances of it failing, whereas the buggy test seemed to fail with regularity: every 2-3 attempts at the least, possibly
more frequently.

## Tentative conclusion	

We suspect that struct members that are USM pointers are not captured correctly in SYCL lambdas. Discussions
with James Brodman indicate that there is such a bug already being worked on. I am putting in this bugreport
so that the issue can be tracked on GitHub also.



## Detailed code structure and description

The clone will create a directory called `parallel_for`

In this directory, 

* the program to exercise the bug is in `main.cpp`
* the Kokkos proxies are in `KokkosProxies.hpp`
* there is a `Makefile`
* the prototype Kokkos back end is in the `parallel_for/external/kokkos`

In `main.cpp` we define a functor as follows:

```
struct functor {

  double mult;
  my_ptr_t ptr_d;

  void operator()(const int i,  cl::sycl::stream out) const {

    if ( ptr_d == 0x0 ) {
      if ( i  == 2 ) {
        out << "Id = " << i << " BARF!!!" << cl::sycl::endl;
      }
    }
    else {
      ((double *)ptr_d)[i] = mult*(double)i+1.5;
    }
  }


  // These are just so the functor could print itself out both in and outside
  // SYCL kernels

  friend std::ostream& operator<<(std::ostream& os, functor const& that)
  { return os << "my addr=" << &that << " mult=" << that.mult << " ptr_d=" << that.ptr_d; }

  friend cl::sycl::stream operator<<(cl::sycl::stream os, functor const& that)
  { return os << "my addr=" << &that << " mult=" << that.mult << " ptr_d=" << that.ptr_d; }

};
```

We then allocate a device pointer using USM for the `ptr_d` member and dispatch this functor
both using the Kokkos back end and Kokkos proxies as follows:

```
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


        functor f;
        f.mult=4.0;
        f.ptr_d = (my_ptr_t)ptr_d;
        Foo::Bar::my_parallel_for_2(N,f);
        copyAndPrint(ptr_d,q,N,f.mult);

        std::cout << "Kokkos::parallel_for " << std::endl;
        f.mult = 6.0;
        f.ptr_d = (my_ptr_t)ptr_d;
        Kokkos::parallel_for(N,f);
        Kokkos::fence();
        copyAndPrint(ptr_d,q,N,f.mult);
...
}

```

The `Foo::Bar::my_parallel_for_2` is a proxy for Kokkos parallel for. The 'copyAndPrint` function
copies the data from the device pointer into a buffer, which we can then print on the host with 
a host accessor.

Our Proxy Parallel for is defined in `KokkosProxies.hpp` as follows

```
namespace Foo {
namespace Bar {

// Proxy Range Policy
struct RangePolicy {
        size_t start_idx;
        size_t num_items;

        RangePolicy( size_t start_idx_, size_t num_items_) : start_idx(start_idx_), num_items(num_items_) {}
        inline
        size_t begin() { return start_idx; }

        // Past last item.
        inline
        size_t end() { return num_items; }
};


// Proxy parallel for closure
template< class FunctorType>
class ParallelFor {
public:
        using Policy = Foo::Bar::RangePolicy;

        FunctorType m_functor;
        Policy m_policy;

        inline
        void execute() const {
                launch(*this);
        }


        ParallelFor( const FunctorType & arg_functor ,
                        const Policy   &     arg_policy )
        : m_functor( arg_functor )
        , m_policy(  arg_policy )
        { }
};

// Proxy Launcher
template<class Driver>
void launch(Driver driver_in) {
        std::cerr << "In sycl_launch_copy" << std::endl;

        // cl::sycl::queue* q = driver_in.m_policy.space().impl_internal_space_instance()->m_queue;
        cl::sycl::queue* q = Kokkos::Experimental::SYCL().impl_internal_space_instance()->m_queue;
        std::cerr << "Queue pointer is: " << (unsigned long) q << std::endl;

        std::cerr << "range=" << driver_in.m_policy.end()-driver_in.m_policy.begin() << std::endl;
        std::cerr << "driver_in.ptr_d = " << (unsigned long)(driver_in.m_functor.ptr_d) << std::endl;

        q->submit([&](cl::sycl::handler& cgh) {
                cl::sycl::stream out(1024,256,cgh);
                cgh.parallel_for (
                                cl::sycl::range<1>(driver_in.m_policy.end()-driver_in.m_policy.begin()),
                                [=] (cl::sycl::id<1> item) {
                        size_t idx = item[0];
                        if (idx == 2 ) { // stop threads overwriting
                                out << "idx = " << idx << " PF ptr_d = " << (unsigned long)driver_in.m_functor.ptr_d << cl::sycl::endl;
                        }
                        driver_in.m_functor(idx, out); // Use passed in functor captured in the template Driver struct
                                                       // which is just the ParallelFor above.
                });
        });
        q->wait_and_throw();
}


// Proxy ParallelFor Dispatch call
template <class FunctorType>
inline void my_parallel_for_2(const size_t work_count, const FunctorType& functor,
                const std::string& str = "") {

        //Foo::Bar::RangePolicy policy(0,work_count);
        Foo::Bar::ParallelFor<FunctorType> closure(functor,
                        Foo::Bar::RangePolicy(0, work_count));


        std::cerr << "Calling execute" <<std::endl;

        closure.execute();
}
```
In contrast the equivalent definitions are in `externals/kokkos/core/src/SYCL`.
The closure is defined in `Kokkos_SYCL_Parallel_Range.hpp`:

```
namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits ... >
                 , Kokkos::Experimental::SYCL
                 >
{
public:

  typedef Kokkos::RangePolicy< Traits ... > Policy;
  FunctorType  m_functor ;
  Policy       m_policy ;

// inline
  void execute() //const
  {
#ifdef NO_LAMBDA
          std::cerr << "in execute: ptr_d =  " << m_functor.ptr_d << std::endl;
#endif
          Kokkos::Experimental::Impl::sycl_launch(*this);
    }

  ParallelFor( const FunctorType & arg_functor ,
               const Policy   &     arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    { }

};

}
}
```
The launcher is in `external/kokkos/core/src/SYCL/Kokkos_SYCL_KernelLaunch.hpp`:
```
#include <SYCL/Kokkos_SYCL_Error.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template<class Driver>
void sycl_launch(Driver driver_in) {
  std::cerr << "In sycl_launch" << std::endl;

  cl::sycl::queue* q = Kokkos::Experimental::SYCL().impl_internal_space_instance()->m_queue;
  std::cerr << "Queue pointer is: " << (unsigned long) q << std::endl;

#ifdef NO_LAMBDA
  std::cerr << "range=" << driver_in.m_policy.end()-driver_in.m_policy.begin() << std::endl;
  std::cerr << "driver_in.ptr_d = " << (unsigned long)(driver_in.m_functor.ptr_d) << std::endl;
  std::cerr << "Functor: " << driver_in.m_functor << std::endl;
#endif

      q->submit([&](cl::sycl::handler& cgh) {
          cl::sycl::stream out(1024,256,cgh);

          auto localfunctor = driver_in.m_functor;  // Make a copy of the passed in functor
                                                    //
#ifdef NO_LAMBDA
          std::cout << "Before par for: ptr_d=" << driver_in.m_functor << std::endl;
          std::cout << "Before par for: local copy = " << localfunctor << std::endl;
#endif
          cgh.parallel_for (
                  cl::sycl::range<1>(driver_in.m_policy.end()-driver_in.m_policy.begin()),
                  [=] (cl::sycl::id<1> item) {
                 size_t idx = item[0];
                 if (idx == 2 ) { // stop threads overwriting
#ifdef NO_LAMBDA
                   out << "In par for: Before Kernel call: idx = " << idx << " argument functor : " <<  driver_in.m_functor << cl::sycl::endl;
                   out << "In par for: Before Kernel call: idx = " << idx << " localcopy : " <<  localfunctor << cl::sycl::endl;
#endif
                 }

                
                if ( idx == 2 ) {  out << "Calling Localfunctor " << cl::sycl::endl; }
                localfunctor(idx,out);

                //    if ( idx == 2 ) { out << "Calling driver_in.m_functor " << cl::sycl::endl; }
                //    driver_in.m_functor(idx,out);
                

         });
      });
      q->wait_and_throw();
}

}}} // Namespace
```
The `#ifdef NO_LAMBDA` can be activated by adding `-DNO_LAMBDA` onto the `TEST_OPTIONS` in the first line of the `Makefile`.
If `-DNO_LAMBDA` is enabled the last test in `main` which launches `Kokkos::parallel_for` using a lambda is disabled and
only the functor class test is run. In this case we can print out the device pointers in the struct, which are not available
otherwise in a lambda functor.

As one can see from the code comments originally we wanted to launch using the passed in `driver_in.m_functor(idx,out)`
call however as also noted the `driver_in.m_functor` when captured in the `cgh.parallel_for` lambda seems to be zeroed
out (although it works fine in my proxy class).




