module stochastic_collect_tau_cam
    ! From Morrison (Lebo, originally TAU bin code)
    ! Gettelman and Chen 2018
    !the subroutines take in air density, air temperature, and the bin mass boundaries, and
    !output the mass and number mixing ratio tendencies in each bin directly.
    !this is then wrapped for CAM.

    ! note, this is now coded locally. Want the CAM interface to be over i,k I think.

    !#ifndef HAVE_GAMMA_INTRINSICS
    !use shr_spfn_mod, only: gamma => shr_spfn_gamma
    !#endif

    !use statements here
    !use

    !use shr_kind_mod,   only: r8=>shr_kind_r8
    !use cam_history,    only: addfld
    !use micro_mg_utils, only: pi, rhow, qsmall

    implicit none
    private
    save

    ! Subroutines
    public :: stochastic_collect_tau_tend



    !In the module top, declare the following so that these can be used throughout the module:

    integer, parameter :: r8 = selected_real_kind(12)
    real(r8), parameter :: pi = 3.14159265358979323846_r8
    real(r8), parameter :: rhow = 1000._r8
    real(r8), parameter :: qsmall = 1.e-18_r8

    integer, parameter, public :: ncd = 35
    integer, parameter, public :: ncdp = ncd + 1
    !integer, parameter, public  :: ncdl = ncd
    !integer, parameter, public  :: ncdpl = ncdl+1

    !integer, private :: ncd,ncdp
    !integer, private :: ncdl,ncdpl
    !PARAMETER(ncd=35,ncdl=ncd) ! set number of ice and liquid bins
    !PARAMETER(ncdp=ncd+1,ncdpl=ncdl+1)

    ! for Zach's collision-coalescence code

    real(r8), private :: knn(ncd, ncd)

    real(r8), public :: mmean(ncd), diammean(ncd)       ! kg & m at bin mid-points
    real(r8), public :: medge(ncdp), diamedge(ncdp)     ! kg & m at bin edges
    integer, private :: cutoff_id                       ! cutoff between cloud water and rain drop, D = 40 microns

    type :: mghydrometeorprops
        ! Density (kg/m^3)
        real(r8) :: rho
        ! Information for size calculations.
        ! Basic calculation of mean size is:
        !     lambda = (shape_coef*nic/qic)^(1/eff_dim)
        ! Then lambda is constrained by bounds.
        real(r8) :: eff_dim
        real(r8) :: shape_coef
        real(r8) :: lambda_bounds(2)
        ! Minimum average particle mass (kg).
        ! Limit is applied at the beginning of the size distribution calculations.
        real(r8) :: min_mean_mass
    end type mghydrometeorprops

    integer, parameter, public :: i8 = selected_int_kind(18)
    integer(i8), parameter :: limiter_off = int(Z'7FF1111111111111', i8)

    interface MGHydrometeorProps
        module procedure NewMGHydrometeorProps
    end interface

    interface rising_factorial
        module procedure rising_factorial_r8
        module procedure rising_factorial_integer
    end interface rising_factorial

    interface size_dist_param_liq
        module procedure size_dist_param_liq_vect
        module procedure size_dist_param_liq_line
    end interface
    interface size_dist_param_basic
        module procedure size_dist_param_basic_vect
        module procedure size_dist_param_basic_line
    end interface

    !===============================================================================
contains
    !===============================================================================


    subroutine calc_bins

        real(r8) :: DIAM(ncdp)
        real(r8) :: X(ncdp)
        real(r8) :: radsl(ncdp)
        real(r8) :: radl(ncd)
        integer :: L, lcl
        real(r8) :: kkfac
        !Then before doing any calculations you'll need to calculate the bin mass grid
        ! (note this code could be cleaned up, I'm just taking it as it's used in our bin scheme).
        ! This only needs to be done once, since we'll use the same bin mass grid for all calculations.

        ! use mass doubling bins from Graham Feingold (note cgs units)

        !      PI=3.14159_r8
        !      rho_w=1000._r8                         ! kg/m3
        DIAM(1) = 1.5625 * 2.E-04_r8                ! cm
        X(1) = PI / 6._r8 * DIAM(1)**3 * rhow / 1000._r8  ! rhow kg/m3 --> g/cm3
        radsl(1) = X(1)                         ! grams
        !      radsl(1) = X(1)/1000._r8

        DO l = 2, ncdp
            X(l) = 2._r8 * X(l - 1)
            DIAM(l) = (6._r8 / pi * X(l) * 1000._r8 / rhow)**(1._r8 / 3._r8)  ! cm
            !         radsl(l)=X(l)/1000._r8 ! convert from g to kg
            radsl(l) = X(l)
        ENDDO

        ! now get bin mid-points

        do l = 1, ncd
            radl(l) = (radsl(l) + radsl(l + 1)) / 2._r8         ! grams
            !         diammean(l) = (DIAM(l)+DIAM(l+1))/2._r8     ! cm
            diammean(l) = (6._r8 / pi * radl(l) * 1000._r8 / rhow)**(1._r8 / 3._r8) ! cm
        end do

        ! set bin grid for method of moments

        ! for method of moments

        do lcl = 1, ncd + 1
            !         medge(lcl) = radsl(lcl)*1000._r8     ! convert to grams
            medge(lcl) = radsl(lcl)               ! grams
            diamedge(lcl) = DIAM(lcl)             ! cm
        enddo

        do lcl = 1, ncd
            !         mmean(lcl) = radl(lcl)*1000._r8
            mmean(lcl) = radl(lcl)
            diammean(lcl) = diammean(lcl)
        enddo

        do lcl = ncdp, 1, -1
            if(diamedge(lcl).ge.40.e-4_r8) cutoff_id = lcl
        end do

    end subroutine calc_bins

    subroutine stochastic_kernel_init

        !    use cam_history_support, only: add_hist_coord

        integer :: idd, jdd
        real(r8) :: kkfac

        call calc_bins


        ! Read in the collection kernel code from a lookup table. Again, this only needs to be done once.
        ! use kernel from Zach (who got it from Jerry)

        KNN(:, :) = 0._r8 ! initialize values
        kkfac = 1.5_r8   ! from Zach
        open(unit = 40, file = '/glade/u/home/cchen/forDJ/v3/KBARF', status = 'old')

        941 FORMAT(2X, E12.5)

        do idd = 1, ncd
            do jdd = 1, idd
                READ(40, 941) KNN(IDD, JDD)
                !     KNN(IDD,JDD)=(XK_GR(IDD)*kkfac+XK_GR(JDD)*kkfac)*KNN(IDD,JDD)
                KNN(IDD, JDD) = (mmean(IDD) * kkfac + mmean(JDD) * kkfac) * KNN(IDD, JDD)

                if (knn(idd, jdd).lt.0._r8) knn(idd, jdd) = 0._r8

            end do
        end do
        close(unit = 40)
    end subroutine stochastic_kernel_init

    !main driver routine
    !needs to pull in i,k fields (so might need dimensions here too)

    subroutine stochastic_collect_tau_tend(deltatin, t, rho, qc, qr, qcin, ncin, qrin, nrin, lcldm, precip_frac, &
            !                                       mu_c, lambda_c, n0r, lambda_r, &
            qcin_new, ncin_new, qrin_new, nrin_new, &
            !                                       qctend,nctend,qrtend,nrtend,qctend_TAU,nctend_TAU,qrtend_TAU,nrtend_TAU, &
            qctend_TAU, nctend_TAU, qrtend_TAU, nrtend_TAU, &
            scale_qc, scale_nc, scale_qr, scale_nr, &
            amk_c, ank_c, amk_r, ank_r, amk, ank, amk_out, ank_out, gmnnn_lmnnn_TAU, mgncol)


        !use micro_mg_utils, only: &
        !       mg_liq_props, &
        !       mg_rain_props

        !use micro_mg_utils, only: &
        !       size_dist_param_liq, &
        !       size_dist_param_basic

        !inputs: mgncol,nlev,t,rho,qcin,ncin,qrin,nrin
        !outputs: qctend,nctend,qrtend,nrtend
        !not sure if we want to output bins (extra dimension). Good for testing?

        integer, intent(in) :: mgncol

        real(r8), intent(in) :: deltatin
        real(r8), intent(in) :: t(mgncol)
        real(r8), intent(in) :: rho(mgncol)
        real(r8), intent(in) :: qc(mgncol)
        real(r8), intent(in) :: qr(mgncol)
        real(r8), intent(in) :: qcin(mgncol)
        real(r8), intent(in) :: ncin(mgncol)
        real(r8), intent(in) :: qrin(mgncol)
        real(r8), intent(in) :: nrin(mgncol)
        real(r8) :: qcic(mgncol)
        real(r8) :: ncic(mgncol)
        real(r8) :: qric(mgncol)
        real(r8) :: nric(mgncol)
        real(r8), intent(in) :: lcldm(mgncol)
        real(r8), intent(in) :: precip_frac(mgncol)
        !real(r8), intent(inout) :: qctend(mgncol)
        !real(r8), intent(inout) :: nctend(mgncol)
        !real(r8), intent(inout) :: qrtend(mgncol)
        !real(r8), intent(inout) :: nrtend(mgncol)
        real(r8), intent(out) :: qctend_TAU(mgncol)
        real(r8), intent(out) :: nctend_TAU(mgncol)
        real(r8), intent(out) :: qrtend_TAU(mgncol)
        real(r8), intent(out) :: nrtend_TAU(mgncol)

        real(r8), intent(out) :: scale_qc(mgncol)
        real(r8), intent(out) :: scale_nc(mgncol)
        real(r8), intent(out) :: scale_qr(mgncol)
        real(r8), intent(out) :: scale_nr(mgncol)

        real(r8), intent(out) :: amk_c(mgncol, ncd)
        real(r8), intent(out) :: ank_c(mgncol, ncd)
        real(r8), intent(out) :: amk_r(mgncol, ncd)
        real(r8), intent(out) :: ank_r(mgncol, ncd)
        real(r8), intent(out) :: amk(mgncol, ncd)
        real(r8), intent(out) :: ank(mgncol, ncd)
        real(r8), intent(out) :: amk_out(mgncol, ncd)
        real(r8), intent(out) :: ank_out(mgncol, ncd)

        real(r8) :: mu_c(mgncol)
        real(r8) :: lambda_c(mgncol)
        real(r8) :: lambda_r(mgncol)
        real(r8) :: n0r(mgncol)

        real(r8) :: gnnnn(ncd)
        real(r8) :: gmnnn(ncd)
        real(r8) :: lnnnn(ncd)
        real(r8) :: lmnnn(ncd)

        real(r8), intent(out) :: qcin_new(mgncol)
        real(r8), intent(out) :: ncin_new(mgncol)
        real(r8), intent(out) :: qrin_new(mgncol)
        real(r8), intent(out) :: nrin_new(mgncol)
        real(r8), intent(out) :: gmnnn_lmnnn_TAU(mgncol)

        real(r8) :: qcin_old(mgncol)
        real(r8) :: ncin_old(mgncol)
        real(r8) :: qrin_old(mgncol)
        real(r8) :: nrin_old(mgncol)

        integer :: i, lcl, cutoff_amk, cutoff(mgncol)

        real(r8) :: all_gmnnn, all_lmnnn

        real(r8), parameter :: dsph = 3._r8
        real(r8), parameter :: min_mean_mass_liq = 1.e-20_r8
        real(r8), parameter :: lam_bnd_rain(2) = 1._r8 / [500.e-6_r8, 20.e-6_r8]
        type(mghydrometeorprops) :: mg_liq_props
        type(mghydrometeorprops) :: mg_rain_props

        call stochastic_kernel_init

        cutoff = cutoff_id - 1

        qcic = qcin
        ncic = ncin
        qric = qrin
        nric = nrin

        mg_liq_props = MGHydrometeorProps(rhow, dsph, &
                min_mean_mass = min_mean_mass_liq)

        mg_rain_props = MGHydrometeorProps(rhow, dsph, lam_bnd_rain)

        call size_dist_param_liq(mg_liq_props, qcic(1:mgncol), ncic(1:mgncol), &
                rho(1:mgncol), mu_c(1:mgncol), lambda_c(1:mgncol), mgncol)

        call size_dist_param_basic(mg_rain_props, qric(:), nric(:), &
                lambda_r(:), mgncol, n0 = n0r(:))

        !do k = 1,nlev
        !call size_dist_param_liq(mg_liq_props, qcin, ncin, rho, mu_c, lambda_c, mgncol)
        !call size_dist_param_basic(mg_rain_props, qrin, nrin, lambda_r, mgncol, n0=n0r)
        !end do

        ! First make bins from cam size distribution (bins are diagnostic)

        !call cam_bin_distribute(qcin,ncin,qrin,nrin,medge,amk,ank)
        do i = 1, mgncol
            !do k=1,nlev
            call cam_bin_distribute(qc(i), qr(i), qcin(i), ncin(i), qrin(i), nrin(i), &
                    mu_c(i), lambda_c(i), lambda_r(i), n0r(i), lcldm(i), precip_frac(i), &
                    scale_qc(i), scale_nc(i), scale_qr(i), scale_nr(i), &
                    amk_c(i, 1:ncd), ank_c(i, 1:ncd), amk_r(i, 1:ncd), ank_r(i, 1:ncd), amk(i, 1:ncd), ank(i, 1:ncd), cutoff_amk)
            !end do
            if(cutoff_amk.gt.0) then
                cutoff(i) = cutoff_amk
            end if
            !   cutoff(i) = cutoff_id-1
        end do

        !Then call the subroutines that actually do the calculations. The inputs/ouputs are described in comments below.

        !This part of the code needs to be called each time for each process rate calculation
        ! (i.e., for each sampled cloud/rain gamma distribution):

        ! note: variables passed to compute_column_params are all inputs,
        ! outputs from this subroutine are stored as global variables

        ! inputs: t --> input air temperature (K)
        !         rho --> input air density (kg/m^3)
        !         medge --> bin mass boundary (g)
        !         amk --> array of bin mass mixing ratio, i.e., the input drop mass distribution (kg/kg)
        !         ank --> array of bin number mixing ratio, i.e., the input drop number distribution (kg^-1)

        ! inputs: medge --> bin mass boundary (g), same as above

        ! outputs: gnnnn --> bin number mass mixing tendency gain, array in bins (#/cm^3/s)
        !          lnnnn --> bin number mass mixing tendency loss, array in bins (#/cm^3/s)
        !          gmnnn --> bin mass mixing ratio tendency gain, array in bins (g/cm^3/s)
        !          lmnnn --> bin mass mixing ratio tendency loss, array in bins (g/cm^3/s)


        ! Call Kernel

        !do i=1,mgncol
        !do k=1,nlev
        !   call do_nn_n(gnnnn(i,:),gmnnn(i,:),lnnnn(i,:),lmnnn(i,:),medge)
        !end do
        !end do

        qcin_new = 0._r8
        ncin_new = 0._r8
        qrin_new = 0._r8
        nrin_new = 0._r8

        qcin_old = 0._r8
        ncin_old = 0._r8
        qrin_old = 0._r8
        nrin_old = 0._r8

        qctend_TAU = 0._r8
        nctend_TAU = 0._r8
        qrtend_TAU = 0._r8
        nrtend_TAU = 0._r8


        ! update qc, nc, qr, nr
        do i = 1, mgncol
            !do k=1,nlev
            !   gnnnn = 0._r8
            !   gmnnn = 0._r8
            !   lnnnn = 0._r8
            !   lmnnn = 0._r8

            !   if( (qc(i).gt.qsmall).or.(qr(i).gt.qsmall) ) then
            call compute_coll_params(rho(i), medge, amk(i, 1:ncd), ank(i, 1:ncd), gnnnn, gmnnn, lnnnn, lmnnn)
            !   call do_nn_n(gnnnn,gmnnn,lnnnn,lmnnn,medge)
            !   end if

            all_gmnnn = 0._r8
            all_lmnnn = 0._r8
            ! scaling gmnnn, lmnnn
            do lcl = 1, ncd
                all_gmnnn = all_gmnnn + gmnnn(lcl)
                all_lmnnn = all_lmnnn + lmnnn(lcl)
            end do

            if((all_gmnnn.eq.0._r8).or.(all_lmnnn.eq.0._r8)) then
                gmnnn(:) = 0._r8
                lmnnn(:) = 0._r8
            else
                lmnnn = lmnnn * (all_gmnnn / all_lmnnn)
            end if
            ! cloud water
            do lcl = 1, cutoff(i)
                qcin_old(i) = qcin_old(i) + amk(i, lcl)
                ncin_old(i) = ncin_old(i) + ank(i, lcl)
                qcin_new(i) = qcin_new(i) + (gmnnn(lcl) - lmnnn(lcl)) * 1.e3_r8 / rho(i) * deltatin
                ncin_new(i) = ncin_new(i) + (gnnnn(lcl) - lnnnn(lcl)) * 1.e6_r8 / rho(i) * deltatin

                qctend_TAU(i) = qctend_TAU(i) + (gmnnn(lcl) - lmnnn(lcl)) * 1.e3_r8 / rho(i)
                nctend_TAU(i) = nctend_TAU(i) + (gnnnn(lcl) - lnnnn(lcl)) * 1.e6_r8 / rho(i)
                gmnnn_lmnnn_TAU(i) = gmnnn_lmnnn_TAU(i) + gmnnn(lcl) - lmnnn(lcl)
            end do
            ! rain
            do lcl = cutoff(i) + 1, ncd
                qrin_old(i) = qrin_old(i) + amk(i, lcl)
                nrin_old(i) = nrin_old(i) + ank(i, lcl)
                qrin_new(i) = qrin_new(i) + (gmnnn(lcl) - lmnnn(lcl)) * 1.e3_r8 / rho(i) * deltatin
                nrin_new(i) = nrin_new(i) + (gnnnn(lcl) - lnnnn(lcl)) * 1.e6_r8 / rho(i) * deltatin

                qrtend_TAU(i) = qrtend_TAU(i) + (gmnnn(lcl) - lmnnn(lcl)) * 1.e3_r8 / rho(i)
                nrtend_TAU(i) = nrtend_TAU(i) + (gnnnn(lcl) - lnnnn(lcl)) * 1.e6_r8 / rho(i)
                gmnnn_lmnnn_TAU(i) = gmnnn_lmnnn_TAU(i) + gmnnn(lcl) - lmnnn(lcl)
            end do

            do lcl = 1, ncd
                amk_out(i, lcl) = amk(i, lcl) + (gmnnn(lcl) - lmnnn(lcl)) * 1.e3_r8 / rho(i) * deltatin
                ank_out(i, lcl) = ank(i, lcl) + (gnnnn(lcl) - lnnnn(lcl)) * 1.e6_r8 / rho(i) * deltatin
            end do

            qcin_new(i) = qcin_new(i) + qcin_old(i)
            ncin_new(i) = ncin_new(i) + ncin_old(i)
            qrin_new(i) = qrin_new(i) + qrin_old(i)
            nrin_new(i) = nrin_new(i) + nrin_old(i)

            !   if( (qcin_new(i)+qrin_new(i)).gt.0._r8 ) then
            !      qctend_TAU(i) = qctend_TAU(i)*((qcin(i)+qrin(i))/(qcin_new(i)+qrin_new(i)))
            !      qrtend_TAU(i) = qrtend_TAU(i)*((qcin(i)+qrin(i))/(qcin_new(i)+qrin_new(i)))
            !   end if

            !   amk_c(i,:) = amk_c(i,:)*lcldm(i)
            !   ank_c(i,:) = ank_c(i,:)*lcldm(i)
            !   amk_r(i,:) = amk_r(i,:)*precip_frac(i)
            !   ank_r(i,:) = ank_r(i,:)*precip_frac(i)
            !   amk = amk_c+amk_r
            !   ank = ank_c+ank_r

            !   qctend_TAU(i) = (qcin_new(i)-qcin(i)*lcldm(i))/deltatin
            !   nctend_TAU(i) = (ncin_new(i)-ncin(i)*lcldm(i))/deltatin
            !   qrtend_TAU(i) = (qrin_new(i)-qrin(i)*precip_frac(i))/deltatin
            !   nrtend_TAU(i) = (nrin_new(i)-nrin(i)*precip_frac(i))/deltatin

            !   qctend_TAU(i) = (qcin_new(i)-qcin_old(i))/deltatin
            !   nctend_TAU(i) = (ncin_new(i)-ncin_old(i))/deltatin
            !   qrtend_TAU(i) = (qrin_new(i)-qrin_old(i))/deltatin
            !   nrtend_TAU(i) = (nrin_new(i)-nrin_old(i))/deltatin

            !   qctend(i) = qctend(i) + (qcic_new(i)-qcic(i)*lcldm(i))/deltatin
            !   nctend(i) = nctend(i) + (ncic_new(i)-ncic(i)*lcldm(i))/deltatin
            !   qrtend(i) = qrtend(i) + (qric_new(i)-qric(i)*precip_frac(i))/deltatin
            !   nrtend(i) = nrtend(i) + (nric_new(i)-nric(i)*precip_frac(i))/deltatin
            !end do
        end do

        ! To calculate the number tendency in each bin you'll need to subtract the loss term from the gain term,
        ! and do a unit conversion to get units of kg^-1:

        !  bin number tendency =  (gnnnn(lcl)-lnnnn(lcl))/rho*1.e6

        !  And the same for the mass tendency in each bin to get units of kg/kg:

        ! bin mass tendency = (gmnnn(lcl)-lmnnn(lcl))/(rho/1000.)



        !In the calculations above "lcl" is just the bin array index.

        ! Then add the bin number and mass up to a certain threshold size to get the bulk mass and number mixing ratio
        ! tendencies for cloud, and everything larger than this threshold to get the bulk mass and number mixing ratio
        ! tendencies for rain.

    end subroutine stochastic_collect_tau_tend


    subroutine cam_bin_distribute(qc_all, qr_all, qc, nc, qr, nr, mu_c, lambda_c, lambda_r, n0r, &
            lcldm, precip_frac, scale_qc, scale_nc, scale_qr, scale_nr, &
            amk_c, ank_c, amk_r, ank_r, amk, ank, cutoff_amk)

        real(r8) :: qc_all, qr_all, qc, nc, qr, nr, mu_c, lambda_c, lambda_r, n0r, lcldm, precip_frac
        real(r8), dimension(ncd) :: amk_c, ank_c, amk_r, ank_r, amk, ank
        integer :: i
        real(r8) :: phi
        real(r8) :: scale_nc, scale_qc, scale_nr, scale_qr

        integer :: id_max_qc, id_max_qr, cutoff_amk
        real(r8) :: max_qc, max_qr, min_amk

        ank_c = 0._r8
        amk_c = 0._r8
        ank_r = 0._r8
        amk_r = 0._r8
        ank = 0._r8
        amk = 0._r8

        scale_nc = 0._r8
        scale_qc = 0._r8
        scale_nr = 0._r8
        scale_qr = 0._r8

        id_max_qc = 0
        id_max_qr = 0
        cutoff_amk = 0
        max_qc = 0._r8
        max_qr = 0._r8

        ! cloud water, nc in #/m3 --> #/cm3
        if((qc_all.gt.qsmall).and.(qc.gt.qsmall)) then
            do i = 1, ncd
                phi = nc * lambda_c**(mu_c + 1._r8) / shr_spfn_gamma(mu_c + 1._r8) * (diammean(i) * 1.e-2_r8)**mu_c * exp(-lambda_c * diammean(i) * 1.e-2_r8) ! D cm --> m
                ank_c(i) = phi * (diamedge(i + 1) - diamedge(i)) * 1.e-2_r8                     ! D cm --> m
                amk_c(i) = phi * (diamedge(i + 1) - diamedge(i)) * 1.e-2_r8 * mmean(i) * 1.e-3_r8   ! mass in bin g --> kg

                scale_nc = scale_nc + ank_c(i)
                scale_qc = scale_qc + amk_c(i)
            end do

            scale_nc = scale_nc / nc
            scale_qc = scale_qc / qc

            ank_c = ank_c / scale_nc * lcldm
            amk_c = amk_c / scale_qc * lcldm
            !ank_c = ank_c*lcldm
            !amk_c = amk_c*lcldm

            do i = 1, ncd
                if(amk_c(i).gt.max_qc) then
                    id_max_qc = i
                    max_qc = amk_c(i)
                end if
            end do

            !else

            !do i=1,ncd
            !   ank_c(i) = 0._r8
            !   amk_c(i) = 0._r8
            !end do

        end if

        ! rain drop
        if((qr_all.gt.qsmall).and.(qr.gt.qsmall)) then
            do i = 1, ncd
                phi = n0r * exp(-lambda_r * diammean(i) * 1.e-2_r8)                   ! D cm --> m
                ank_r(i) = phi * (diamedge(i + 1) - diamedge(i)) * 1.e-2_r8    ! D cm --> m
                amk_r(i) = phi * (diamedge(i + 1) - diamedge(i)) * 1.e-2_r8 * mmean(i) * 1.e-3_r8

                scale_nr = scale_nr + ank_r(i)
                scale_qr = scale_qr + amk_r(i)
            end do

            scale_nr = scale_nr / nr
            scale_qr = scale_qr / qr

            ank_r = ank_r / scale_nr * precip_frac
            amk_r = amk_r / scale_qr * precip_frac
            !ank_r = ank_r*precip_frac
            !amk_r = amk_r*precip_frac

            !else

            !do i=1,ncd
            !   ank_r(i) = 0._r8
            !   amk_r(i) = 0._r8
            !end do

            do i = 1, ncd
                if(amk_r(i).gt.max_qr) then
                    id_max_qr = i
                    max_qr = amk_r(i)
                end if
            end do

        end if

        amk = amk_c + amk_r
        ank = ank_c + ank_r

        if((id_max_qc.gt.0).and.(id_max_qr.gt.0)) then
            if((max_qc / max_qr.lt.10._r8).or.(max_qc / max_qr.gt.0.1_r8))then
                min_amk = amk(id_max_qc)

                do i = id_max_qc, id_max_qr
                    if(amk(i).le.min_amk) then
                        cutoff_amk = i
                        min_amk = amk(i)
                    end if
                end do
            end if
        end if


        !if( qc_all.gt.qsmall.OR.qr_all.gt.qsmall ) then
        !   do i=1,ncd
        !      ank(i) = ank_c(i) + ank_r(i)
        !      amk(i) = amk_c(i) + amk_r(i)
        !   end do
        !else
        !   do i=1,ncd
        !      amk(i) = 0._r8
        !      ank(i) = 0._r8
        !   end do
        !end if
        !input: qc,nc,qr,nr, medge (bin edges). May also need # bins?
        !output: amk, ank (mixing ratio and number in each bin)

        !this part will take a bit of thinking about.
        !use size distribution parameters (mu, lambda) to generate the values at discrete size points
        !need to also ensure mass conservation

    end subroutine cam_bin_distribute


    ! here are the subroutines called above that actually do the collision-coalescence calculations:

    ! The Kernel is from Jerry from many moons ago (included)

    ! I read in the file data and multiply by the summed mass of the individual bins
    ! (with a factor of 1.5 so that the values represent the middle of the bin

    ! 941 FORMAT(2X,E12.5)
    !     READ(40,941) KNN(IDD,JDD)
    !     KNN(IDD,JDD)=(XK_GR(IDD)*kkfac+XK_GR(JDD)*kkfac)*KNN(IDD,JDD)

    !where idd and jdd are the indexes for the bins and xk_gr is the mass of drops in a bin in grams
    !

    !************************************************************************************
    ! Setup variables needed for collection
    ! Either pass in or define globally the following variables
    ! tbase(height) - temperature in K as a function of height
    ! rhon(height) - air density as a function of height in kg/m^3
    ! xk_gr(bins) - mass of single drop in each bin in grams
    ! lsmall - small number
    ! QC - mass mixing ratio in kg/kg
    ! QN - number mixing ratio in #/kg
    ! All parameters are defined to be global in my version so that they are readily available throughout the code:
    ! SMN0,SNN0,SMCN,APN,AMN2,AMN3,PSIN,FN,FPSIN,XPSIN,HPSIN,FN2,XXPSIN (all arrays of drop bins)
    !************************************************************************************

    !AG: Global arrays need to be passed around I think? Right now at the module level. Is that okay?

    SUBROUTINE COMPUTE_COLL_PARAMS(rhon, xk_gr, qc, qn, gnnnn, gmnnn, lnnnn, lmnnn)
        IMPLICIT NONE

        ! variable declarations (added by hm, 020118)
        ! note: vertical array looping is stripped out, this subroutine operates
        ! only on LOCAL values

        real(r8), dimension(ncd) :: qc, qn
        real(r8), dimension(ncdp) :: xk_gr
        real(r8) :: tbase, rhon
        !  real(r8) :: TAIRC,UMMS,UMMS2
        integer :: lk
        integer :: l
        real(r8), parameter :: lsmall = 1.e-12_r8
        real(r8), dimension(ncd) :: smn0, snn0, smcn, amn2, amn3, psin, fn, fpsin, &
                xpsin, hpsin, fn2, xxpsin
        real(r8) :: apn

        real(r8), dimension(ncd) :: gnnnn, gmnnn, lnnnn, lmnnn
        integer :: lm1, ll

        lk = ncd



        !....................................................................................
        !  TAIRC=TBASE(K)-273.15
        !  TAIRC=TBASE-273.15_r8
        !  UMMS=UMM(TAIRC)
        !!  UMMS2=UMMS*4.66/(RHON(K)/1.E3)
        !!  UMMS=UMMS/(RHON(K)/1.E3)
        !  UMMS2=UMMS*4.66_r8/(RHON/1.E3_r8)
        !  UMMS=UMMS/(RHON/1.E3_r8)

        DO L = 1, LK
            !     SMN0(L)=QC(L,K)*RHON(K)/1.E3
            !     SNN0(L)=QN(L,K)*RHON(K)/1.E6
            SMN0(L) = QC(L) * RHON / 1.E3_r8
            SNN0(L) = QN(L) * RHON / 1.E6_r8

            IF(SMN0(L).LT.lsmall.OR.SNN0(L).LT.lsmall)THEN
                SMN0(L) = 0.0_r8
                SNN0(L) = 0.0_r8
            ENDIF
        ENDDO

        DO L = 1, LK
            IF(SMN0(L) .gt. 0._r8.AND.SNN0(L) .gt. 0._r8)THEN
                SMCN(L) = SMN0(L) / SNN0(L)
                IF((SMCN(L) .GT. 2._r8 * XK_GR(L)))THEN
                    !           SMCN(L) = (2*XK_GR(L))
                    SMCN(L) = (2._r8 * XK_GR(L))
                ENDIF
                IF((SMCN(L) .LT. XK_GR(L)))THEN
                    SMCN(L) = XK_GR(L)
                ENDIF
            ELSE
                SMCN(L) = 0._r8
            ENDIF
            IF (SMCN(L).LT.XK_GR(L).OR.SMCN(L).GT.(2._r8 * XK_GR(L)).OR.SMCN(L).EQ.0.0_r8)THEN
                APN = 1.0_r8
            ELSE
                !        APN=0.5*(1.+3.*(XK_GR(L)/SMCN(L))-2*((XK_GR(L)/SMCN(L))**2.))
                APN = 0.5_r8 * (1._r8 + 3._r8 * (XK_GR(L) / SMCN(L)) - 2._r8 * ((XK_GR(L) / SMCN(L))**2._r8))
            ENDIF

            IF(SNN0(L) .GT. LSMALL)THEN
                AMN2(L) = APN * SMN0(L) * SMN0(L) / SNN0(L)
                AMN3(L) = APN * APN * APN * SMN0(L) * SMN0(L) * SMN0(L) / (SNN0(L) * SNN0(L))
            ELSE
                AMN2(L) = 0._r8
                AMN3(L) = 0._r8
            ENDIF

            IF(SMCN(L).LT.XK_GR(L))THEN
                PSIN(L) = 0.0_r8
                FN(L) = 2._r8 * SNN0(L) / XK_GR(L)
            ELSE
                IF(SMCN(L).GT.(2._r8 * XK_GR(L)))THEN
                    FN(L) = 0.0_r8
                    PSIN(L) = 2._r8 * SNN0(L) / XK_GR(L)
                ELSE
                    PSIN(L) = 2._r8 / XK_GR(L) * (SMN0(L) / XK_GR(L) - SNN0(L))
                    FN(L) = 2._r8 / XK_GR(L) * (2._r8 * SNN0(L) - SMN0(L) / XK_GR(L))
                ENDIF
            ENDIF

            IF(SNN0(L).LT.LSMALL.OR.SMN0(L).LT.LSMALL)THEN
                PSIN(L) = 0.0_r8
                FN(L) = 0.0_r8
            ENDIF

            FPSIN(L) = 0.5_r8 / XK_GR(L) * (PSIN(L) - FN(L))
            XPSIN(L) = 2._r8 * XK_GR(L) * PSIN(L)
            HPSIN(L) = PSIN(L) - 0.5_r8 * FN(L)
            FN2(L) = FN(L) / 2._r8

            IF(L.GT.1)THEN
                XXPSIN(L) = XK_GR(L) * PSIN(L - 1)
            ENDIF
        ENDDO

        !************************************************************************************
        ! Compute collision coalescence
        ! Either pass in or define globally the following variables
        ! Gain terms begin with G, loss terms begin with L
        ! Second letter defines mass (M) or number (N)
        ! Third and fourth letters define the types of particles colling, i.e., NN means drops colliding with drops
        ! Last letter defines the category the new particles go into, in this case just N for liquid drops
        ! The resulting rates are in units of #/cm^3/s and g/cm^3/s
        ! Relies on predefined kernel array KNN(bins,bins) - see top of this file
        !************************************************************************************

        GMNNN = 0._r8
        GNNNN = 0._r8
        LMNNN = 0._r8
        LNNNN = 0._r8
        ! remove verical array index, calculate gain/loss terms locally

        DO L = 3, LK - 1
            LM1 = L - 1
            DO LL = 1, L - 2
                !        GNNNN(L,K)=GNNNN(L,K)+(PSIN(LM1)*SMN0(LL)-FPSIN(LM1)*AMN2(LL))*KNN(LM1,LL)
                !        GMNNN(L,K)=GMNNN(L,K)+(XK_GR(L)*PSIN(LM1)*SMN0(LL)+FN2(LM1)*AMN2(LL)-FPSIN(LM1)*AMN3(LL))*KNN(LM1,LL)
                GNNNN(L) = GNNNN(L) + (PSIN(LM1) * SMN0(LL) - FPSIN(LM1) * AMN2(LL)) * KNN(LM1, LL)
                GMNNN(L) = GMNNN(L) + (XK_GR(L) * PSIN(LM1) * SMN0(LL) + FN2(LM1) * AMN2(LL) - FPSIN(LM1) * AMN3(LL)) * KNN(LM1, LL)
            ENDDO
        ENDDO

        DO L = 2, LK - 1
            LM1 = L - 1
            GNNNN(L) = GNNNN(L) + 0.5_r8 * SNN0(LM1) * SNN0(LM1) * KNN(LM1, LM1)
            GMNNN(L) = GMNNN(L) + 0.5_r8 * (SNN0(LM1) * SMN0(LM1) + SMN0(LM1) * SNN0(LM1)) * KNN(LM1, LM1)
            DO LL = 1, L - 1
                !        LNNNN(L,K)=LNNNN(L,K)+(PSIN(L)*SMN0(LL)-FPSIN(L)*AMN2(LL))*KNN(L,LL)
                !        GMNNN(L,K)=GMNNN(L,K)+(SMN0(LL)*SNN0(L)-PSIN(L)*AMN2(LL)+FPSIN(L)*AMN3(LL))*KNN(L,LL)
                !        LMNNN(L,K)=LMNNN(L,K)+(XPSIN(L)*SMN0(LL)-HPSIN(L)*AMN2(LL))*KNN(L,LL)
                LNNNN(L) = LNNNN(L) + (PSIN(L) * SMN0(LL) - FPSIN(L) * AMN2(LL)) * KNN(L, LL)
                GMNNN(L) = GMNNN(L) + (SMN0(LL) * SNN0(L) - PSIN(L) * AMN2(LL) + FPSIN(L) * AMN3(LL)) * KNN(L, LL)
                LMNNN(L) = LMNNN(L) + (XPSIN(L) * SMN0(LL) - HPSIN(L) * AMN2(LL)) * KNN(L, LL)
            ENDDO
        ENDDO

        DO L = 1, LK - 1
            DO LL = L, LK - 1
                !        LNNNN(L,K)=LNNNN(L,K)+(SNN0(LL)*SNN0(L))*KNN(LL,L)
                !        LMNNN(L,K)=LMNNN(L,K)+(SNN0(LL)*SMN0(L))*KNN(LL,L)
                LNNNN(L) = LNNNN(L) + (SNN0(LL) * SNN0(L)) * KNN(LL, L)
                LMNNN(L) = LMNNN(L) + (SNN0(LL) * SMN0(L)) * KNN(LL, L)
            ENDDO
        ENDDO

    END SUBROUTINE COMPUTE_COLL_PARAMS


    function shr_spfn_gamma(x) result(gamma)

        real(r8), parameter :: sqrtpi = 1.77245385090551602729_r8
        ! Machine epsilon
        real(r8), parameter :: epsr8 = epsilon(1._r8)
        ! "Huge" value is returned when actual value would be infinite.
        real(r8), parameter :: xinfr8 = huge(1._r8)
        ! Smallest normal value.
        real(r8), parameter :: xminr8 = tiny(1._r8)
        ! Largest number that, when added to 1., yields 1.
        real(r8), parameter :: xsmallr8 = epsr8 / 2._r8
        ! Largest argument for which erfcx > 0.
        real(r8), parameter :: xmaxr8 = 1._r8 / (sqrtpi * xminr8)
        ! For gamma/igamma
        ! Approximate value of largest acceptable argument to gamma,
        ! for IEEE double-precision.
        real(r8) :: xbig_gamma = 171.624_r8

        real(r8), intent(in) :: x
        real(r8) :: gamma
        real(r8) :: fact, res, sum, xden, xnum, y, y1, ysq, z

        integer :: i, n
        logical :: negative_odd

        ! log(2*pi)/2
        real(r8), parameter :: logsqrt2pi = 0.9189385332046727417803297E0_r8

        !----------------------------------------------------------------------
        !  NUMERATOR AND DENOMINATOR COEFFICIENTS FOR RATIONAL MINIMAX
        !     APPROXIMATION OVER (1,2).
        !----------------------------------------------------------------------
        real(r8), parameter :: P(8) = &
                (/-1.71618513886549492533811E+0_r8, 2.47656508055759199108314E+1_r8, &
                        -3.79804256470945635097577E+2_r8, 6.29331155312818442661052E+2_r8, &
                        8.66966202790413211295064E+2_r8, -3.14512729688483675254357E+4_r8, &
                        -3.61444134186911729807069E+4_r8, 6.64561438202405440627855E+4_r8 /)
        real(r8), parameter :: Q(8) = &
                (/-3.08402300119738975254353E+1_r8, 3.15350626979604161529144E+2_r8, &
                        -1.01515636749021914166146E+3_r8, -3.10777167157231109440444E+3_r8, &
                        2.25381184209801510330112E+4_r8, 4.75584627752788110767815E+3_r8, &
                        -1.34659959864969306392456E+5_r8, -1.15132259675553483497211E+5_r8 /)
        !----------------------------------------------------------------------
        !  COEFFICIENTS FOR MINIMAX APPROXIMATION OVER (12, INF).
        !----------------------------------------------------------------------
        real(r8), parameter :: C(7) = &
                (/-1.910444077728E-03_r8, 8.4171387781295E-04_r8, &
                        -5.952379913043012E-04_r8, 7.93650793500350248E-04_r8, &
                        -2.777777777777681622553E-03_r8, 8.333333333333333331554247E-02_r8, &
                        5.7083835261E-03_r8 /)

        negative_odd = .false.
        fact = 1._r8
        n = 0
        y = x
        if (y <= 0._r8) then
            !----------------------------------------------------------------------
            !  ARGUMENT IS NEGATIVE
            !----------------------------------------------------------------------
            y = -x
            y1 = aint(y)
            res = y - y1
            if (res /= 0._r8) then
                negative_odd = (y1 /= aint(y1 * 0.5_r8) * 2._r8)
                fact = -pi / sin(pi * res)
                y = y + 1._r8
            else
                gamma = xinfr8
                return
            end if
        end if
        !----------------------------------------------------------------------
        !  ARGUMENT IS POSITIVE
        !----------------------------------------------------------------------
        if (y < epsr8) then
            !----------------------------------------------------------------------
            !  ARGUMENT .LT. EPS
            !----------------------------------------------------------------------
            if (y >= xminr8) then
                res = 1._r8 / y
            else
                gamma = xinfr8
                return
            end if
        elseif (y < 12._r8) then
            y1 = y
            if (y < 1._r8) then
                !----------------------------------------------------------------------
                !  0.0 .LT. ARGUMENT .LT. 1.0
                !----------------------------------------------------------------------
                z = y
                y = y + 1._r8
            else
                !----------------------------------------------------------------------
                !  1.0 .LT. ARGUMENT .LT. 12.0, REDUCE ARGUMENT IF NECESSARY
                !----------------------------------------------------------------------
                n = int(y) - 1
                y = y - real(n, r8)
                z = y - 1._r8
            end if
            !----------------------------------------------------------------------
            !  EVALUATE APPROXIMATION FOR 1.0 .LT. ARGUMENT .LT. 2.0
            !----------------------------------------------------------------------
            xnum = 0._r8
            xden = 1._r8
            do i = 1, 8
                xnum = (xnum + P(i)) * z
                xden = xden * z + Q(i)
            end do
            res = xnum / xden + 1._r8
            if (y1 < y) then
                !----------------------------------------------------------------------
                !  ADJUST RESULT FOR CASE  0.0 .LT. ARGUMENT .LT. 1.0
                !----------------------------------------------------------------------
                res = res / y1
            elseif (y1 > y) then
                !----------------------------------------------------------------------
                !  ADJUST RESULT FOR CASE  2.0 .LT. ARGUMENT .LT. 12.0
                !----------------------------------------------------------------------
                do i = 1, n
                    res = res * y
                    y = y + 1._r8
                end do
            end if
        else
            !----------------------------------------------------------------------
            !  EVALUATE FOR ARGUMENT .GE. 12.0,
            !----------------------------------------------------------------------
            if (y <= xbig_gamma) then
                ysq = y * y
                sum = C(7)
                do i = 1, 6
                    sum = sum / ysq + C(i)
                end do
                sum = sum / y - y + logsqrt2pi
                sum = sum + (y - 0.5_r8) * log(y)
                res = exp(sum)
            else
                gamma = xinfr8
                return
            end if
        end if
        !----------------------------------------------------------------------
        !  FINAL ADJUSTMENTS AND RETURN
        !----------------------------------------------------------------------
        if (negative_odd)  res = -res
        if (fact /= 1._r8) res = fact / res
        gamma = res
        ! ---------- LAST LINE OF GAMMA ----------
    end function shr_spfn_gamma

    ! get cloud droplet size distribution parameters
    elemental subroutine size_dist_param_liq_line(props, qcic, ncic, rho, pgam, lamc)
        type(MGHydrometeorProps), intent(in) :: props
        real(r8), intent(in) :: qcic
        real(r8), intent(inout) :: ncic
        real(r8), intent(in) :: rho

        real(r8), intent(out) :: pgam
        real(r8), intent(out) :: lamc

        type(MGHydrometeorProps) :: props_loc

        if (qcic > qsmall) then

            ! Local copy of properties that can be modified.
            ! (Elemental routines that operate on arrays can't modify scalar
            ! arguments.)
            props_loc = props

            ! Get pgam from fit to observations of martin et al. 1994
            pgam = 1.0_r8 - 0.7_r8 * exp(-0.008_r8 * 1.e-6_r8 * ncic * rho)
            pgam = 1._r8 / (pgam**2) - 1._r8
            pgam = max(pgam, 2._r8)

            ! Set coefficient for use in size_dist_param_basic.
            ! The 3D case is so common and optimizable that we specialize it:
            if (props_loc%eff_dim == 3._r8) then
                props_loc%shape_coef = pi / 6._r8 * props_loc%rho * &
                        rising_factorial(pgam + 1._r8, 3)
            else
                props_loc%shape_coef = pi / 6._r8 * props_loc%rho * &
                        rising_factorial(pgam + 1._r8, props_loc%eff_dim)
            end if

            ! Limit to between 2 and 50 microns mean size.
            props_loc%lambda_bounds = (pgam + 1._r8) * 1._r8 / [50.e-6_r8, 2.e-6_r8]

            call size_dist_param_basic(props_loc, qcic, ncic, lamc)

        else
            ! pgam not calculated in this case, so set it to a value likely to
            ! cause an error if it is accidentally used
            ! (gamma function undefined for negative integers)
            pgam = -100._r8
            lamc = 0._r8
        end if

    end subroutine size_dist_param_liq_line

    ! get cloud droplet size distribution parameters

    subroutine size_dist_param_liq_vect(props, qcic, ncic, rho, pgam, lamc, mgncol)

        type(mghydrometeorprops), intent(in) :: props
        integer, intent(in) :: mgncol
        real(r8), dimension(mgncol), intent(in) :: qcic
        real(r8), dimension(mgncol), intent(inout) :: ncic
        real(r8), dimension(mgncol), intent(in) :: rho
        real(r8), dimension(mgncol), intent(out) :: pgam
        real(r8), dimension(mgncol), intent(out) :: lamc
        type(mghydrometeorprops) :: props_loc
        integer :: i

        do i = 1, mgncol
            if (qcic(i) > qsmall) then
                ! Local copy of properties that can be modified.
                ! (Elemental routines that operate on arrays can't modify scalar
                ! arguments.)
                props_loc = props
                ! Get pgam from fit to observations of martin et al. 1994
                pgam(i) = 1.0_r8 - 0.7_r8 * exp(-0.008_r8 * 1.e-6_r8 * ncic(i) * rho(i))
                pgam(i) = 1._r8 / (pgam(i)**2) - 1._r8
                pgam(i) = max(pgam(i), 2._r8)
            endif
        enddo
        do i = 1, mgncol
            if (qcic(i) > qsmall) then
                ! Set coefficient for use in size_dist_param_basic.
                ! The 3D case is so common and optimizable that we specialize
                ! it:
                if (props_loc%eff_dim == 3._r8) then
                    props_loc%shape_coef = pi / 6._r8 * props_loc%rho * &
                            rising_factorial(pgam(i) + 1._r8, 3)
                else
                    props_loc%shape_coef = pi / 6._r8 * props_loc%rho * &
                            rising_factorial(pgam(i) + 1._r8, props_loc%eff_dim)
                end if
                ! Limit to between 2 and 50 microns mean size.
                props_loc%lambda_bounds(1) = (pgam(i) + 1._r8) * 1._r8 / 50.e-6_r8
                props_loc%lambda_bounds(2) = (pgam(i) + 1._r8) * 1._r8 / 2.e-6_r8
                call size_dist_param_basic(props_loc, qcic(i), ncic(i), lamc(i))
            endif
        enddo
        do i = 1, mgncol
            if (qcic(i) <= qsmall) then
                ! pgam not calculated in this case, so set it to a value likely to
                ! cause an error if it is accidentally used
                ! (gamma function undefined for negative integers)
                pgam(i) = -100._r8
                lamc(i) = 0._r8
            end if
        enddo

    end subroutine size_dist_param_liq_vect

    ! Basic routine for getting size distribution parameters.
    elemental subroutine size_dist_param_basic_line(props, qic, nic, lam, n0)
        type(MGHydrometeorProps), intent(in) :: props
        real(r8), intent(in) :: qic
        real(r8), intent(inout) :: nic

        real(r8), intent(out) :: lam
        real(r8), intent(out), optional :: n0

        if (qic > qsmall) then

            ! add upper limit to in-cloud number concentration to prevent
            ! numerical error
            if (limiter_is_on(props%min_mean_mass)) then
                nic = min(nic, qic / props%min_mean_mass)
            end if

            ! lambda = (c n/q)^(1/d)
            lam = (props%shape_coef * nic / qic)**(1._r8 / props%eff_dim)

            ! check for slope
            ! adjust vars
            if (lam < props%lambda_bounds(1)) then
                lam = props%lambda_bounds(1)
                nic = lam**(props%eff_dim) * qic / props%shape_coef
            else if (lam > props%lambda_bounds(2)) then
                lam = props%lambda_bounds(2)
                nic = lam**(props%eff_dim) * qic / props%shape_coef
            end if

        else
            lam = 0._r8
        end if

        if (present(n0)) n0 = nic * lam

    end subroutine size_dist_param_basic_line

    subroutine size_dist_param_basic_vect(props, qic, nic, lam, mgncol, n0)

        type (mghydrometeorprops), intent(in) :: props
        integer, intent(in) :: mgncol
        real(r8), dimension(mgncol), intent(in) :: qic
        real(r8), dimension(mgncol), intent(inout) :: nic
        real(r8), dimension(mgncol), intent(out) :: lam
        real(r8), dimension(mgncol), intent(out), optional :: n0
        integer :: i
        do i = 1, mgncol

            if (qic(i) > qsmall) then

                ! add upper limit to in-cloud number concentration to prevent
                ! numerical error
                if (limiter_is_on(props%min_mean_mass)) then
                    nic(i) = min(nic(i), qic(i) / props%min_mean_mass)
                end if

                ! lambda = (c n/q)^(1/d)
                lam(i) = (props%shape_coef * nic(i) / qic(i))**(1._r8 / props%eff_dim)

                ! check for slope
                ! adjust vars
                if (lam(i) < props%lambda_bounds(1)) then
                    lam(i) = props%lambda_bounds(1)
                    nic(i) = lam(i)**(props%eff_dim) * qic(i) / props%shape_coef
                else if (lam(i) > props%lambda_bounds(2)) then
                    lam(i) = props%lambda_bounds(2)
                    nic(i) = lam(i)**(props%eff_dim) * qic(i) / props%shape_coef
                end if

            else
                lam(i) = 0._r8
            end if

        enddo

        if (present(n0)) n0 = nic * lam

    end subroutine size_dist_param_basic_vect


    function NewMGHydrometeorProps(rho, eff_dim, lambda_bounds, min_mean_mass) &
            result(res)
        real(r8), intent(in) :: rho, eff_dim
        real(r8), intent(in), optional :: lambda_bounds(2), min_mean_mass
        type(mghydrometeorprops) :: res

        res%rho = rho
        res%eff_dim = eff_dim
        if (present(lambda_bounds)) then
            res%lambda_bounds = lambda_bounds
        else
            res%lambda_bounds = no_limiter()
        end if
        if (present(min_mean_mass)) then
            res%min_mean_mass = min_mean_mass
        else
            res%min_mean_mass = no_limiter()
        end if

        res%shape_coef = rho * pi * gamma(eff_dim + 1._r8) / 6._r8

    end function NewMGHydrometeorProps

    pure function rising_factorial_r8(x, n) result(res)
        real(r8), intent(in) :: x, n
        real(r8) :: res

        res = gamma(x + n) / gamma(x)

    end function rising_factorial_r8

    ! Rising factorial can be performed much cheaper if n is a small integer.
    pure function rising_factorial_integer(x, n) result(res)
        real(r8), intent(in) :: x
        integer, intent(in) :: n
        real(r8) :: res

        integer :: i
        real(r8) :: factor

        res = 1._r8
        factor = x

        do i = 1, n
            res = res * factor
            factor = factor + 1._r8
        end do

    end function rising_factorial_integer


    pure function no_limiter()
        real(r8) :: no_limiter

        no_limiter = transfer(limiter_off, no_limiter)

    end function no_limiter

    pure function limiter_is_on(lim)
        real(r8), intent(in) :: lim
        logical :: limiter_is_on

        limiter_is_on = transfer(lim, limiter_off) /= limiter_off

    end function limiter_is_on


end module stochastic_collect_tau_cam


