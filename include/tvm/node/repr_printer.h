/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file tvm/node/repr_printer.h
 * \brief Printer class to print repr string of each AST/IR nodes.
 */
#ifndef TVM_NODE_REPR_PRINTER_H_
#define TVM_NODE_REPR_PRINTER_H_

#include <tvm/node/functor.h>
#include <tvm/node/script_printer.h>

#include <iostream>
#include <optional>
#include <string>

namespace tvm {
/*! \brief A printer class to print the AST/IR nodes. */
class ReprPrinter {
 public:
  /*! \brief The output stream */
  std::ostream& stream;
  /*! \brief The indentation level. */
  int indent{0};

  explicit ReprPrinter(std::ostream& stream)  // NOLINT(*)
      : stream(stream) {}

  /*! \brief The node to be printed. */
  TVM_DLL void Print(const ObjectRef& node);
  /*! \brief Print indent to the stream */
  TVM_DLL void PrintIndent();
  // Allow registration to be printer.
  using FType = NodeFunctor<void(const ObjectRef&, ReprPrinter*)>;
  TVM_DLL static FType& vtable();
};

/*! \brief Legacy behavior of ReprPrinter. */
class ReprLegacyPrinter {
 public:
  /*! \brief The indentation level. */
  int indent{0};

  explicit ReprLegacyPrinter(std::ostream& stream)  // NOLINT(*)
      : stream(stream) {}

  /*! \brief The node to be printed. */
  TVM_DLL void Print(const ObjectRef& node);
  /*! \brief Print indent to the stream */
  TVM_DLL void PrintIndent();
  /*! \brief Could the LegacyPrinter dispatch the node */
  TVM_DLL static bool CanDispatch(const ObjectRef& node);
  /*! \brief Return the ostream it maintains */
  TVM_DLL std::ostream& Stream() const;
  // Allow registration to be printer.
  using FType = NodeFunctor<void(const ObjectRef&, ReprLegacyPrinter*)>;
  TVM_DLL static FType& vtable();

 private:
  /*! \brief The output stream */
  std::ostream& stream;
};

/*!
 * \brief Dump the node to stderr, used for debug purposes.
 * \param node The input node
 */
TVM_DLL void Dump(const runtime::ObjectRef& node);

/*!
 * \brief Dump the node to stderr, used for debug purposes.
 * \param node The input node
 */
TVM_DLL void Dump(const runtime::Object* node);

TVM_DLL std::string AsRepr(const runtime::ObjectRef& node);

}  // namespace tvm

namespace tvm {
namespace runtime {
// default print function for all objects
// provide in the runtime namespace as this is where objectref originally comes from.
inline std::ostream& operator<<(std::ostream& os, const ObjectRef& n) {  // NOLINT(*)
  ReprPrinter(os).Print(n);
  return os;
}

inline std::string AsLegacyRepr(const ObjectRef& n) {
  std::ostringstream os;
  ReprLegacyPrinter(os).Print(n);
  return os.str();
}
}  // namespace runtime
using runtime::AsLegacyRepr;
}  // namespace tvm

namespace std {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& n) {
  os << "[";
  for (size_t i = 0; i < n.size(); ++i) {
    if (i != 0) os << ", ";
    os << n[i];
  }
  os << "]";
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& n) {
  return (n.has_value() ? (os << n.value()) : (os << "nullopt"));
}

template <typename K, typename V, typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::unordered_map<K, V, T1, T2>& n) {
  os << "{";
  for (auto it = n.begin(); it != n.end(); ++it) {
    if (it != n.begin()) os << ", ";
    os << it->first << ": " << it->second;
  }
  os << "}";
  return os;
}

template <typename K, typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<K, T1, T2>& n) {
  os << "{";
  for (auto it = n.begin(); it != n.end(); ++it) {
    if (it != n.begin()) os << ", ";
    os << *it;
  }
  os << "}";
  return os;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& n) {
  os << "(" << n.first << ", " << n.second << ")";
  return os;
}

template <typename... Ts, size_t... Is>
void PrintTuple(std::ostream& os, const std::tuple<Ts...>& n, std::index_sequence<Is...>) {
  os << "(";
  ((os << std::get<Is>(n) << ", "), ...);
  os << std::get<sizeof...(Ts) - 1>(n) << ")";
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& n) {
  if (sizeof...(Ts) == 0) {
    os << "()";
    return os;
  }
  PrintTuple(os, n, std::make_index_sequence<sizeof...(Ts) - 1>{});
  return os;
}
}  // namespace std

#endif  // TVM_NODE_REPR_PRINTER_H_
