// A less-specialized version of
// https://github.com/joboccara/NamedType/blob/master/named_type_impl.hpp
// customized for elastica needs
/*
MIT License

Copyright (c) 2017 Jonathan Boccara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#pragma once

#include <type_traits>
#include <utility>

namespace utilities {

  template <typename T, typename Parameter, template <typename> class... Skills>
  class NamedType : public Skills<NamedType<T, Parameter, Skills...>>... {
   public:
    using UnderlyingType = T;

    // constructor
    explicit constexpr NamedType(T const& value) : value_(value) {}
    // Works for a reference T too
    template <typename T_ = T, typename std::enable_if<
                                   !std::is_reference<T_>::value, void>::type>
    explicit constexpr NamedType(T&& value) : value_(std::move(value)) {}

    // get
    constexpr T& get() { return value_; }
    constexpr std::remove_reference_t<T> const& get() const { return value_; }

    // conversions
    using ref = NamedType<T&, Parameter, Skills...>;
    operator ref() { return ref(value_); }

   private:
    T value_;
  };

}  // namespace utilities
