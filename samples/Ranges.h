/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <Eigen/Dense>

#include <array>
#include <memory>
#include <vector>

namespace gauss::gp::samples {
class Linspace {
public:
  Linspace(const double min, const double max, const std::size_t size);

  Linspace(const Linspace &o);

  bool operator()() const { return i < size; };
  double eval() const { return min + delta * i; };
  std::size_t index() const { return i; };

  Linspace &operator++();

private:
  const double min;
  const double delta;
  std::size_t i;
  const std::size_t size;
};

std::vector<Eigen::VectorXd> linspace(const double min, const double max,
                                      const std::size_t size);

class Grid {
public:
  Grid(const std::array<double, 2> &min_corner,
       const std::array<double, 2> &max_corner, const std::size_t size);

  bool operator()() const { return row() && (*col)(); };
  Eigen::Vector2d eval() const;
  std::array<std::size_t, 2> indices() const {
    return {row.index(), col->index()};
  }

  Grid &operator++();

private:
  Linspace row;
  std::unique_ptr<Linspace> col;
};

std::vector<std::vector<Eigen::Vector2d>>
grid(const std::array<double, 2> &min_corner,
     const std::array<double, 2> &max_corner, const std::size_t size);
} // namespace gauss::gp::samples
