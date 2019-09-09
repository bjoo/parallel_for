/*
 * KokkosProxies.hpp
 *
 *  Created on: Sep 9, 2019
 *      Author: bjoo
 */

#ifndef KOKKOSPROXIES_HPP_
#define KOKKOSPROXIES_HPP_

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

} // namespace Bar
} // Namespace Foo



#endif /* KOKKOSPROXIES_HPP_ */
