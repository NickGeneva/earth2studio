! SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
! SPDX-FileCopyrightText: All rights reserved.
! SPDX-License-Identifier: Apache-2.0

!> ISO_C_BINDING wrapper around NCEPLIBS-bufr for earth2bufrio.
!!
!! Exposes a small set of C-callable functions that the Python ctypes
!! layer can call to read BUFR / PrepBUFR files.
module earth2bufrio_fort
  use iso_c_binding
  implicit none

  ! Internal state for Fortran unit allocation
  integer, save :: next_unit = 20

contains

  !> Open a BUFR file and return a Fortran logical unit number.
  !!
  !! The caller passes a C character array and its length.
  !! Returns the allocated unit number (>0) on success, or -1 on failure.
  integer(c_int) function e2b_open(filepath, filepath_len) bind(c, name='e2b_open')
    character(c_char), intent(in) :: filepath(*)
    integer(c_int), value, intent(in) :: filepath_len

    character(len=512) :: fpath
    integer :: lun, i

    ! Copy C string to Fortran string
    fpath = ' '
    do i = 1, min(filepath_len, 512)
      fpath(i:i) = filepath(i)
    end do

    ! Allocate a unit number
    lun = next_unit
    next_unit = next_unit + 1

    ! Set 10-digit date format
    call datelen(10)

    ! Open the file
    open(unit=lun, file=trim(fpath), status='old', form='unformatted', &
         access='sequential', iostat=i)
    if (i /= 0) then
      e2b_open = -1_c_int
      return
    end if

    ! Initialize BUFR reading
    call openbf(lun, 'IN', lun)

    e2b_open = int(lun, c_int)
  end function e2b_open


  !> Read the next BUFR message.
  !!
  !! Returns 0 on success (message available), 1 on EOF / no more messages.
  integer(c_int) function e2b_next_message(lun, msg_type, msg_type_len, idate) &
      bind(c, name='e2b_next_message')
    integer(c_int), value, intent(in) :: lun
    character(c_char), intent(out) :: msg_type(*)
    integer(c_int), intent(out) :: msg_type_len
    integer(c_int), intent(out) :: idate

    character(len=8) :: subset
    integer :: jdate, iret, i

    call readmg(int(lun), subset, jdate, iret)
    if (iret /= 0) then
      e2b_next_message = 1_c_int
      msg_type_len = 0_c_int
      idate = 0_c_int
      return
    end if

    ! Copy subset name to output
    do i = 1, 8
      msg_type(i) = subset(i:i)
    end do
    msg_type_len = int(len_trim(subset), c_int)
    idate = int(jdate, c_int)
    e2b_next_message = 0_c_int
  end function e2b_next_message


  !> Read the next subset within the current message.
  !!
  !! Returns 0 on success, 1 when no more subsets remain.
  integer(c_int) function e2b_next_subset(lun) bind(c, name='e2b_next_subset')
    integer(c_int), value, intent(in) :: lun
    integer :: iret

    call readsb(int(lun), iret)
    if (iret /= 0) then
      e2b_next_subset = 1_c_int
    else
      e2b_next_subset = 0_c_int
    end if
  end function e2b_next_subset


  !> Read scalar or multi-level values for a mnemonic (wraps ufbint).
  !!
  !! Returns 0 on success.  nvalues is set to the number of values read.
  integer(c_int) function e2b_read_values(lun, mnemonic, mnem_len, &
      values, max_values, nvalues) bind(c, name='e2b_read_values')
    integer(c_int), value, intent(in) :: lun
    character(c_char), intent(in) :: mnemonic(*)
    integer(c_int), value, intent(in) :: mnem_len
    real(c_double), intent(out) :: values(*)
    integer(c_int), value, intent(in) :: max_values
    integer(c_int), intent(out) :: nvalues

    character(len=80) :: mnem_str
    real(8) :: buf(255)
    integer :: n, i

    mnem_str = ' '
    do i = 1, min(int(mnem_len), 80)
      mnem_str(i:i) = mnemonic(i)
    end do

    n = 0
    call ufbint(int(lun), buf, 1, min(int(max_values), 255), n, trim(mnem_str))

    nvalues = int(n, c_int)
    do i = 1, n
      values(i) = buf(i)
    end do

    e2b_read_values = 0_c_int
  end function e2b_read_values


  !> Read replicated values for a mnemonic (wraps ufbrep).
  !!
  !! Returns 0 on success.  nvalues is set to the number of values read.
  integer(c_int) function e2b_read_replicated(lun, mnemonic, mnem_len, &
      values, max_values, nvalues) bind(c, name='e2b_read_replicated')
    integer(c_int), value, intent(in) :: lun
    character(c_char), intent(in) :: mnemonic(*)
    integer(c_int), value, intent(in) :: mnem_len
    real(c_double), intent(out) :: values(*)
    integer(c_int), value, intent(in) :: max_values
    integer(c_int), intent(out) :: nvalues

    character(len=80) :: mnem_str
    real(8) :: buf(255)
    integer :: n, i

    mnem_str = ' '
    do i = 1, min(int(mnem_len), 80)
      mnem_str(i:i) = mnemonic(i)
    end do

    n = 0
    call ufbrep(int(lun), buf, 1, min(int(max_values), 255), n, trim(mnem_str))

    nvalues = int(n, c_int)
    do i = 1, n
      values(i) = buf(i)
    end do

    e2b_read_replicated = 0_c_int
  end function e2b_read_replicated


  !> Close the BUFR file and release the unit.
  subroutine e2b_close(lun) bind(c, name='e2b_close')
    integer(c_int), value, intent(in) :: lun

    call closbf(int(lun))
    close(int(lun))
  end subroutine e2b_close


  !> Return the BUFR missing-value sentinel.
  real(c_double) function e2b_get_bmiss() bind(c, name='e2b_get_bmiss')
    real(8) :: getbmiss
    e2b_get_bmiss = real(getbmiss(), c_double)
  end function e2b_get_bmiss

end module earth2bufrio_fort
