# Lambda Capture Bug

## Introduction

This bug demonstrates a lambda capture bug we found with the Public Intel SYCL compiler and USM.
The bug is as follows: sometimes a device pointer allocated by `malloc_device` is not 
captured correctly. In this example this happens when the pointer is accessed in a *functor* struct
or in a *lambda*, specifically in the prototype Kokkos SYCL Back end dispatch launcher.

Things however seem to work better in some proxy classes, which mimic the Kokkos classes.
Below we show how to clone and build the code, how the bug manifests and how we have been able
to work around it occasionally.

## Getting the example

This example contains the prototype Kokkos back end as a sub-module. Correspondingly
it should be cloned recursively:

> git clone --recursive https://github.com/bjoo/parallel_for.git

## Code structure and description

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
  double* ptr_d;   // This will be a device pointer

  void operator()(const int i,  cl::sycl::stream out) const {

    if ( ptr_d == 0x0 ) { 
      if ( i  == 2 ) {   // So only 1 thread prints
        out << "Id = " << i << " BARF!!!" << cl::sycl::endl;
      }
    }
    else {
      ptr_d[i] = mult*(double)i+1.5; // The actual work
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
                        driver_in.m_functor(idx, out);
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

#ifdef DO_LOCALCOPY
      bool localcopy = true;
#else
      bool localcopy = false;
#endif

#ifdef NO_LAMBDA
      bool printfunctor=true;
#else
      bool printfunctor=false;
#endif
      q->submit([&](cl::sycl::handler& cgh) {
          cl::sycl::stream out(1024,256,cgh);

          auto localfunctor = driver_in.m_functor;
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

                 if ( localcopy  )  {
                    if ( idx == 2 ) {  out << "Calling Localfunctor " << cl::sycl::endl; }
                    localfunctor(idx,out);

                 }
                 else {
                    if ( idx == 2 ) { out << "Calling driver_in.m_functor " << cl::sycl::endl; }
                    driver_in.m_functor(idx,out);
                 }

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

In this launcher we can launch either the functor that was captured in main into the `driver_in` function argument, in which case
the bug can manifest. A workaround is to copy the `driver_in.m_functor` into a local variable in Command Group Scope (after the
`q.submit(` but before the `cgh.parallel_for`. The local copy functor is dispatched if `-DDO_LOCALCOPY` is defined in the `TEST_OPTIONS`
in the `Makefile`, otherwise the original `driver_in.m_functor` is dispatched. We also print a bunch of debug information.



## Compilers and drivers

We need a version of the compiler with USM extensions, and a driver capable of 
USM. I use the following version of the Public SYCL Compiler:

> clang version 9.0.0 (https://github.com/intel/llvm.git 51a6204c09f8c8868cb9675c1a7f4c1386eb3c65)
> Target: x86_64-unknown-linux-gnu
> Thread model: posix
> InstalledDir: /dist/sfw/ubuntu/intel-llvm9-sycl-2019-09-17/bin

and I use the Intel HD Compute driver (NEO) version `19.36.14103`


## Manifesting the bug

Set the first line of the `Makefile` as:

> TEST_OPTIONS=-DDO_LOCALCOPY -DNO_LAMBDA

then build and run the test

> make clean ; make -j 8 
> ./bytes_and_flops.host

In my example the first run went wrong and the second run went right even using the 
local copy functor:

Snippet of first output with annotation by me in C++ style comments ( `// BJ:`)
```
Device 0 : Intel(R) Gen9 HD Graphics NEO Driver Version: 19.36.14103 Is accelerator: YES
 q ptr is : 43160448
 ptr_d is : 33062912
 Calling q.submit() 
Calling execute
In sycl_launch_copy
Queue pointer is: 43160448
range=15
driver_in.ptr_d = 33062912
idx = 2 PF ptr_d = 33062912    // BJ: proxy results using factor = 4 are correct below
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
Kokkos::parallel_for 
in execute: ptr_d =  0x1f88000
In sycl_launch
Queue pointer is: 43160448
range=15
driver_in.ptr_d = 33062912
Functor: my addr=0x7ffe90516440 mult=6 ptr_d=0x1f88000                                             // Kokkos::parallel for with mult=6
Before par for: ptr_d=my addr=0x7ffe90516440 mult=6 ptr_d=0x1f88000                                // BJ: driver_in.m_functor ptr_d is OK  
Before par for: local copy = my addr=0x7ffe90515f40 mult=6 ptr_d=0x1f88000                         // BJ: ptd_d in local copy is OK
In par for: Before Kernel call: idx = 2 argument functor : my addr=0x5093320 mult=0 ptr_d=0x0      // BJ: driver_in.m_functor ptr_d = 0x0
In par for: Before Kernel call: idx = 2 localcopy : my addr=0x5093348 mult=6 ptr_d=0xffffffff01f88000  // BJ: local copy ptr_d is weird
Calling Localfunctor 
0 1.500000
1 5.500000    // BJ: Should be 7.5
2 9.500000    // BJ: Should be 13.5
3 19.500000
4 25.500000

5 31.500000
6 37.500000
7 43.500000
8 49.500000
9 55.500000
10 61.500000
11 67.500000
12 49.500000
13 53.500000
14 85.500000
```

Already we can see that the device pointer `ptr_d` before the parallel for looks correct, both When I ran this a second time the output was correct:
```
driver_in.ptr_d = 19881984    // BJ: proxy parallel for with mult=4 is good below
idx = 2 PF ptr_d = 19881984   
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
Kokkos::parallel_for 
in execute: ptr_d =  0x12f6000
In sycl_launch
Queue pointer is: 29979520
range=15
driver_in.ptr_d = 19881984
Functor: my addr=0x7ffcb8f27a00 mult=6 ptr_d=0x12f6000                             // BJ: Kokkos parallel for with mult=6 now
Before par for: ptr_d=my addr=0x7ffcb8f27a00 mult=6 ptr_d=0x12f6000                // BJ: driver_in.m_functor.ptr_d is good                        
Before par for: local copy = my addr=0x7ffcb8f27500 mult=6 ptr_d=0x12f6000         // BJ: local copy is good
In par for: Before Kernel call: idx = 2 argument functor : my addr=0x4402320 mult=0 ptr_d=0x0 // BJ: in parallel for driver_in.m_functor.ptr_d=0x0
In par for: Before Kernel call: idx = 2 localcopy : my addr=0x4402348 mult=6 ptr_d=0x12f6000  // BJ: localcopy.ptr_d looks GOOD!!
Calling Localfunctor 
0 1.500000 
1 7.500000    // BJ:  Answer is now correct
2 13.500000   // BJ:  Answer is now correct
3 19.500000
4 25.500000
5 31.500000
6 25.500000
7 29.500000
8 49.500000
9 37.500000
10 41.500000
11 67.500000
12 73.500000
13 79.500000
14 57.500000 // BJL 
```
As we could see in both cases the original `driver_in.m_functor.ptr_d` was not captured correctly and was in fact 0. 
In the first instance the `localfunctor.ptr_d` was also bad, but in the second case it was ok.

We now get to the original bug, which is when we dispatch the original `driver_in.m_functor`. We already know this will be 
bad since the `ptr_d` is zero. However, this does not go wrong in the proxy class. To make the example use the origina
`driver_in.m_functor` remove the `-DDO_LOCALCOPY`from the `Makefile` test options like so:

> TEST_OPTIONS=-DNO_LAMBDA

The output in my case looks like so:

```
Queue pointer is: 19231312
range=15
driver_in.ptr_d = 19251200
idx = 2 PF ptr_d = 19251200  // BJ: Original functor gives good resuts
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
Kokkos::parallel_for 
in execute: ptr_d =  0x125c000
In sycl_launch
Queue pointer is: 19231312
range=15
driver_in.ptr_d = 19251200
Functor: my addr=0x7ffd232976d0 mult=6 ptr_d=0x125c000
Before par for: ptr_d=my addr=0x7ffd232976d0 mult=6 ptr_d=0x125c000
Before par for: local copy = my addr=0x7ffd232971d0 mult=6 ptr_d=0x125c000
In par for: Before Kernel call: idx = 2 argument functor : my addr=0x4369320 mult=0 ptr_d=0x0  // BJ: captured driver_in.m_functor.ptr_d is 0
In par for: Before Kernel call: idx = 2 localcopy : my addr=0x4369348 mult=6 ptr_d=0x125c000
Calling driver_in.m_functor 
Id = 2 BARF!!!                 // BJ: captured driver_in.m_functor.ptr_d is 0 so kernel barfs.
0 1.500000                     // BJ: results are unchanged from first (working) result from proxy classes
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
```

James Brodman suggested passing the functor as a pointer. I tried to get this to work but when I did I never got good behavior.
I tried capturing `f` as a pointer in the `Kokkos::parallel_for` like so:
```
	
        f->mult = 6.0;
        f->ptr_d = ptr_d;
        {
          Kokkos::parallel_for(N,f); // f is now a pointer
          Kokkos::fence();
        }
        copyAndPrint(ptr_d,q,N);
```

But this will just record a pointer in the `ParallelFor<FunctionType,Policy>` closure class
(and `FunctionType` will be `functor*`)

In the launcher I now dispatch as:
```
      q->submit([&](cl::sycl::handler& cgh) {
          cl::sycl::stream out(1024,256,cgh);

          auto localfunctor = *(driver_in.m_functor);
#ifdef NO_LAMBDA
          std::cout << "Before par for: ptr_d=" << *(driver_in.m_functor) << std::endl;
          std::cout << "Before par for: local copy = " << (localfunctor) << std::endl;
#endif
          cgh.parallel_for (
                  cl::sycl::range<1>(driver_in.m_policy.end()-driver_in.m_policy.begin()),
                  [=] (cl::sycl::id<1> item) {
                 size_t idx = item[0];
                 auto localfunctor2 = *(driver_in.m_functor);
                 if (idx == 2 ) { // stop threads overwriting

                   out << "In par for: Before Kernel call: idx = " << idx << " argument functor : " <<  *(driver_in.m_functor) << cl::sycl::endl;
                   out << "In par for: Before Kernel call: idx = " << idx << " localcopy : " <<  (localfunctor) << cl::sycl::endl;
                 }

                 if ( idx == 2 ) { out << "Calling driver_in.m_functor " << cl::sycl::endl; }
                 localfunctor2(idx,out);

         });
      });
      q->wait_and_throw();
```

The output from the `Kokkos::parallel_for` is now:



