#pragma once

/*
 *https://www.fluentcpp.com/2018/01/09/strong-templates/
 */

namespace utilities {
  template <typename T, typename Tag>
  class NamedTemplate {
    using ValueType = T;
  };
}  // namespace utilities
