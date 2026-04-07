# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(earth2bufr_fort Fortran C)

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
option(NCEPLIBS_BUFR_SRC_DIR
       "Path to NCEPLIBS-bufr source tree"
       "${CMAKE_CURRENT_SOURCE_DIR}/nceplibs-bufr")

# ------------------------------------------------------------------
# Build NCEPLIBS-bufr as a static library
# ------------------------------------------------------------------
if(EXISTS "${NCEPLIBS_BUFR_SRC_DIR}/CMakeLists.txt")
  add_subdirectory(${NCEPLIBS_BUFR_SRC_DIR} nceplibs-bufr-build EXCLUDE_FROM_ALL)
  set(BUFR_LIB bufr_4)
else()
  # Fall back: assume libbufr_4 is already installed
  find_library(BUFR_LIB NAMES bufr_4 bufr REQUIRED)
endif()

# ------------------------------------------------------------------
# Build the wrapper shared library
# ------------------------------------------------------------------
add_library(earth2bufr_fort SHARED earth2bufr_fort.f90)
target_link_libraries(earth2bufr_fort PRIVATE ${BUFR_LIB})

# Position-independent code for shared library
set_target_properties(earth2bufr_fort PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod"
)

# Install rule (used by Makefile cp step)
install(TARGETS earth2bufr_fort
        LIBRARY DESTINATION lib)
