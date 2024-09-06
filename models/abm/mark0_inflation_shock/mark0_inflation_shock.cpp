
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <algorithm>
#include <random>
#include <numeric>

#include "utils.hpp"
using namespace std;

#define ZERO            1.e-10
#define DOANTICIPATE    true
#define TAU_TAR_NOT_CONST true
#define CB_ACTIVITY true

char dividends = 'A';


double red(double x){ return ( fabs(x) < ZERO ? 0.0 : x);}

double compute_ema(double previous, double current,  double omega=0.2){
  return omega*current + (1-omega)*previous;
}

template<typename T, typename A>
T sum_vector(std::vector<T,A> const& data ){
    T sum = 0.0;

    for (typename std::vector<T,A>::const_iterator it = data.begin(); it != data.end(); ++it){
        sum += *it;
    }
    return sum;
}

double bernouilli_trial(std::mt19937 &gen){
  std::bernoulli_distribution distrib(0.5);

  if (distrib(gen)){
      return 1.0;
  }
  else{
    return -1.0;
  }
}


int main (int argc, char *argv[ ]){

    FILE    *out;
    char    add[10000], name[10000];


    //variables
    double  bust, Pavg, Pold, u, S, Atot, firm_savings, debt_tot, Ytot, Wtot, e, Wavg, inflation, k, propensity, Dtot, rho, Ctot, HHindex;
    double  rhol, rhod, rp_avg, pi_avg, Gamma, u_avg, rm_avg;
    int     N;

    //parameters
    double   gammap, gammaw, theta, alpha_g, eta0m, eta0p, beta, R0, r, alpha, rho0, f, alpha_e, alpha_i, phi_pi, Gamma0, G0, phi, omega, omega_i, delta, gamma_e, delta_p, gamma_cb, theta_cb, phi_pi_init, pi_p, omega_cb, kappa_h, delta_e, delta_w;

    //others;
    double  Pmin, rp, rw, ren, Pnorm, arg, pay_roll, dY, p , tmp, budget, interests, total_sales, shock_st, tmp_p; //energy_profit
    double  Wmax, wage_norm, u_share, deftot;
    int     i, t, seed, new_firm, new_len;
    double  profits;

    double pi_target, e_target; //target inflation and employment for CB
    int    negative_count=0;
    double tau_r , tau_tar, tau_tar_init;
    double wage_factor, gamma_p;



    f = 0.5; // is this c_0
    beta = 2.0; // Price sensitivity parameter
    r = 1.0; 

    double y0 = 0.5;
                     //
    double cfactor = 0.0; // Factor by which consumption goes down during consumption shock.

    double zeta = 0.0; // Productivity factor
    double kappa;
    double zfactor, zfactor_tmp;

    int helico;
    int extra_cons;
    int adapt;
    int cbon; 
    // Preparing the program

    //Parse the arguments passed to program
    if(argc!=56){
        printf("Incorrect input structure %d \n", argc);
        exit(1);
    }

    sscanf(argv[1],  "%lf", &R0); // default = 2.0 ;
    sscanf(argv[2],  "%lf", &theta); // default = 3.0; 
    sscanf(argv[3],  "%lf", &Gamma0); // default = 0.0;
    sscanf(argv[4],  "%lf", &rho0); // default = 0.0 ;
    sscanf(argv[5],  "%lf", &alpha); // default = 0.0;
    sscanf(argv[6],  "%lf", &phi_pi); // default = 0.0;
    sscanf(argv[7],  "%lf", &alpha_e); // default = 0.0;
    sscanf(argv[8],  "%lf", &pi_target); // default = 0.0;
    sscanf(argv[9],  "%lf", &e_target); // default = 0.0;
    sscanf(argv[10],  "%lf", &tau_tar); // default = 0.0;
    sscanf(argv[11],  "%lf", &wage_factor); // default = 1.0;
    sscanf(argv[12],  "%lf", &y0); // default = 0.5;
    sscanf(argv[13],  "%lf", &gammap); // default = 0.01;
    sscanf(argv[14],  "%lf", &eta0m); // default = 0.2;
    sscanf(argv[15],  "%lf", &tau_r); // default = 0.0;
    sscanf(argv[16],  "%lf", &alpha_g); // default = 0.0;
    sscanf(argv[17],  "%lf", &zeta); // default = 1.0;
    sscanf(argv[18],  "%lf", &kappa); // default = 1.25;
    sscanf(argv[19],  "%lf", &G0); // default = 0.5;
    sscanf(argv[20],  "%lf", &phi); // default = 0.1;
    sscanf(argv[21],  "%lf", &omega); // default = 0.2;
    sscanf(argv[22],  "%lf", &delta); // default = 0.02;
    sscanf(argv[23],  "%lf", &gamma_e); // default = 1.0;
    sscanf(argv[24],  "%lf", &delta_p); // default = 0.02;
    sscanf(argv[25],  "%lf", &alpha_i); // default = 1.0;
    sscanf(argv[26],  "%lf", &omega_i); // default = 0.2;
    sscanf(argv[27],  "%lf", &gamma_cb); // default = 5;
    sscanf(argv[28],  "%lf", &theta_cb); // default = 0.002;
    sscanf(argv[29],  "%lf", &gamma_p); // default = 0.002;
    sscanf(argv[30],  "%lf", &shock_st); // default =  1;
    sscanf(argv[31],  "%lf", &delta_e); // default =  0.04;


    VectorPtr ptr_vec = {&R0, &theta, &Gamma0, &rho0, &alpha, &phi_pi, &alpha_e, &pi_target, &e_target, &tau_tar, &wage_factor,&y0, &gammap, &eta0m, &tau_r, &alpha_g, &cfactor, &zeta, &zfactor, &kappa, &G0, &phi, &omega, &delta};


    int shockflag, t_start, t_end, policy_start, policy_end, extra_steps, tsim, Teq, tprint, renorm, tprod_shock, tprod, price_start, price_end;
    
    sscanf(argv[32],  "%lf", &zfactor); // default = 0.5;
    sscanf(argv[33],  "%lf", &cfactor); // default = 0.5;
    sscanf(argv[34],  "%d", &seed); // default =  1; 
    sscanf(argv[35],  "%d", &shockflag); // default = 0; 
    sscanf(argv[36],  "%d", &t_start); // default =  2000;  
    sscanf(argv[37],  "%d", &t_end); // default =  2005;  
    sscanf(argv[38],  "%d", &policy_start); // default =  2000;  
    sscanf(argv[39],  "%d", &policy_end); // default =  2005; 
    sscanf(argv[40],  "%d", &helico); // default =  0.0;  
    sscanf(argv[41],  "%d", &N); // default =  10000;  
    sscanf(argv[42],  "%d", &extra_cons); // default =  0.0;  
    sscanf(argv[43],  "%d", &adapt); // default =  0.0;  
    sscanf(argv[44],  "%d", &extra_steps); // default =  1.0;  
    sscanf(argv[45],  "%d", &tsim); // default = 7000;  
    sscanf(argv[46],  "%d", &Teq); // default =  200;  
    sscanf(argv[47],  "%d", &tprint); // default =  1;  
    sscanf(argv[48],  "%d", &renorm); // default =  1;  
    sscanf(argv[49],  "%d", &cbon); // default =  0; 
    sscanf(argv[50],  "%d", &tprod); // default =  0; 
    sscanf(argv[51],  "%d", &price_start); // default =  2000;  
    sscanf(argv[52],  "%d", &price_end); // default =  2000;
    sscanf(argv[53],  "%lf", &kappa_h); // default =  0.2; 
    sscanf(argv[54],  "%lf", &delta_w); // default =  0.08;  
    sscanf(argv[55], "%s",  add);

    std::vector<int * > ptr_vec_int = { &seed, &shockflag, &t_start, &t_end, &policy_start, &policy_end,&helico , &N, &extra_cons, &adapt, &extra_steps, &tsim, &Teq, &tprint, &renorm, &cbon} ;

    tprod_shock = t_end + tprod;
    zfactor_tmp = zfactor;

    String path_name = "output/";
    ShockDetails shock_details;
    try{
        shock_details = ShockDetails(shockflag, t_start, t_end, tsim);

    } catch (const std::invalid_argument& e){
        std::cerr << "exception: " << e.what() << " Flag supplied was " << shockflag << std::endl;
        std::cout << "Accepted flags from 0-4" << std::endl;
        exit(1);
    }

    String s_add(add);

    String final_file_name = s_add;
    printf("File names %s %s\n", s_add.c_str(), final_file_name.c_str());

    FileWriter output (final_file_name + ".txt");
    Vector output_vector;
    String underscore = std::string("_");


    Shocks debt_policy(policy_start, policy_end, tsim);
    if (shockflag <= 2 || shockflag > 4 ){
        debt_policy.no_shock();
    }

    Shocks price_shock(price_start, price_end, tsim);
    if (price_start == price_end){
        price_shock.no_shock();
    }

    // Define new shocks
    vector<double> cfactor_array{0.15, 0.12, 0.09, 0.06, 0.03};
    vector<double> zfactor_array{ 0.15, 0.12, 0.09, 0.09, 0.09, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995};
    //shock_st=1;
    for(i=0;i<5;i++){
        cfactor_array[i] = 1- shock_st * cfactor_array[i];
        zfactor_array[i] = 1- shock_st * zfactor_array[i];
    }

    // Price Shock

    // ema of price shock
    vector<double> pfactor_array{-0.09 , -0.162,
                                -0.115, -0.077, -0.046, -0.022, -0.003,  0.013, 0.025,  0.035,  0.043, 0.05,
                                 0.055,  0.059,  0.062,  0.065, 0.067,  0.068,  0.07 ,  0.071, 0.072,  0.072,
                                 0.073,  0.073, 0.074,  0.074,  0.074,  0.074,
                                 0.046, 0.024,  0.006, -0.008, -0.019, -0.028, -0.036};
    int counter = 0;    
    int counter_p = 0; 
    double ipp=1;
    double delta_e_0 = delta_e;


    //array
    Vector  P(N), Y(N), D(N), A(N), W(N), PROFITS(N); //energy_profit(N);
    std::vector<int>     ALIVE(N), new_list(N);
    Vector etaplus(N), etaminus(N); // to store hiring and firing rates
    Vector wage_nominal(N);
    //Vector firm_interest(N), firm_interest_t_1(N);
    Vector real_consumption(N); // real consumption of products of firms 

    //double R0 = R;
    eta0p  = R0*eta0m;
    //gammaw = 0.01;
    gammaw = r*gammap;
    double G = G0;

    // Save initial tau_tar
    tau_tar_init = tau_tar;
    phi_pi_init = phi_pi;


    printf("R = %.2e\nN = %d\ntheta = %.2e\ngp = %.2e\tgw = %.2e\nf = %.2e\nb = %.2e\nalphag = %.2e\nalpha=%f\n\n",R0,N,theta,gammap,gammaw,f,beta,alpha_g,alpha);
    printf("rho0 = %.2e\tap = %.2e\tae = %.2e\n",rho0,phi_pi,alpha_e);
    printf("pit = %.2e\tet = %.2e\n",pi_target,e_target);
    printf("taut = %.2e\ttaum = %.2e\n",tau_tar,tau_r);
    printf("eta0m= %.2e\teta0p = %.2e\n",eta0m,eta0p);
    printf("seed %d\n",seed);
    printf("omega_i = %.2e\t alpha_i = %.2e \n",omega_i, alpha_i);
    printf("kappa_h = %.2e\t delta_e = %.2e \t delta_w = %.2e\n",kappa_h, delta_e, delta_w);

    char params[10000];
    sprintf(params,"%.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e",rho0,phi_pi,alpha_e,pi_target,e_target,theta,R0,alpha_g,Gamma0,alpha,tau_r,tau_tar);



    int     collapse, avg_counter;

    double theta0 = theta;
    double alpha_g_0 = alpha_g;
    double gamma_e_0 = gamma_e;
    double zeta0 = zeta;
    double Gammatemp = Gamma0; 
    double energy_sector = 0.0;
    /* ************************************** INIT ************************************** */

    if(seed==-1)
        seed = time( NULL );


    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    Pavg = 0.;
    Ytot = 0.;
    Ctot = 0.;
    Wtot = 0.;
    Atot = 0.;
    HHindex = 0.;
    Wmax = Wavg = 1.;
    Pold = 1.;
    inflation = pi_avg = 0.;
    rhol = rho = rm_avg = rho0;
    rhod = rp_avg = 0.;
    u_avg = 0.;
    pi_p = 0.;
    omega_cb = 1; // Instantaneous CB reaction to price shock
     /* *********************************** INIT  ************************************ */
    for(i=0;i<N;i++){

        ALIVE[i] =  1;
        P[i]     =  1.  + 0.1*(2*dis(gen)-1.);
        Y[i]     =  y0  + 0.1*(2*dis(gen)-1.);
        Y[i]     *= zeta;
        D[i]     =  y0;
        W[i]     =  zeta;
        PROFITS[i] = P[i]*std::min(D[i],Y[i]) - W[i]*Y[i];
        real_consumption[i] = std::min(D[i],Y[i]);
        wage_nominal[i] = W[i];

        A[i] =  2.0*Y[i]*W[i]*dis(gen); // cash balance

        Atot += A[i];
        Ytot += Y[i];
        Ctot += real_consumption[i];
        //Pavg += P[i]*Y[i];
        Pavg += P[i] * real_consumption[i] ;
        Wtot += Y[i]*W[i];

        etaplus[i] = 0.0;
        etaminus[i] = 0.0;
        //energy_profit[i] = 0.0;

    }

    // Profit due to energy price increase
    //energy_profit = 0.;
    
    e = Ytot / N;
    e /= zeta;
    u = 1. - e ;

    //Pavg /= Ytot;
    Pavg /= Ctot;
    Wavg = Wtot/Ytot;
    double p_e = Pavg;
    for(i=0;i<N;i++){
        //A[i]  = A[i]*(N*a0)/tmp;
        A[i]  -= gamma_e*p_e*Y[i];
        //Atot += A[i];
        PROFITS[i] -= gamma_e*p_e*Y[i];
        energy_sector += gamma_e*p_e*Y[i];
        P[i] += gamma_e * p_e;
    }
    //S += delta_e*energy_sector;
    S = e*N;
    //S = 0.0;
    double a0 = 0.0;
    //fix total amount of money to 0
    double M0 = a0*N;
    tmp = Atot + S + energy_sector;
    S = S*(N*a0)/tmp;
    energy_sector = energy_sector*(N*a0)/tmp;
    Atot=0.;

    for(i=0;i<N;i++){
        A[i]  = A[i]*(N*a0)/tmp;
        //A[i]  -= gamma_e*p_e*Y[i];
        Atot += A[i];
        //PROFITS[i] -= gamma_e*p_e*Y[i];
        //energy_sector += gamma_e*p_e*Y[i];
        //P[i] += gamma_e * p_e;
    }

    


    std::cout << "S = " << S << " Atot = " << Atot  <<std::endl;

    std::cout << "zeta " << zeta << " zfactor " << zfactor << " extra_steps " << extra_steps << " G0 " << G0 << " phi " << phi << " omega " << omega << " delta " << delta <<  std::endl;


    double w_nominal = Wavg;
    double product_p = 1.0;

     /* *********************************** END INIT ************************************ */

    /* *********************************** MAIN CYCLE ************************************ */

    bust=0.;

    if( phi_pi < ZERO )
    {
        pi_target = 0.0;
        tau_tar   = 0.0;
    }

    int true_end = 0;



    for(t=0;t<tsim;t++){

        
        //std::cout << t << ", Beginning Pavg: " << Pavg << std::endl;
        //renormalize in unit of price
        if(renorm == 1){
            for(i=0;i<N;i++){
                P[i]/=Pavg;
                W[i]/=Pavg;
                A[i]/=Pavg;
                PROFITS[i] /= Pavg;
            }

            S    /= Pavg;
            p_e /= Pavg;
            energy_sector/=Pavg;
            Wavg /= Pavg;
            Wmax /= Pavg;
            Pold /= Pavg;
            M0 /= Pavg;
            Pavg  = 1.;

        }

        






        /* *********************************** UPDATE ************************************ */

        pi_avg = omega*inflation + (1.-omega)*pi_avg;
        rp_avg = omega*rhod + (1.-omega)*rp_avg;
        rm_avg = omega*rhol + (1.-omega)*rm_avg;
        u_avg = omega*u + (1.-omega)*u_avg;

        // Profit due to energy price increase
        int price_shock_int = price_shock.shock_array[t];


        //update firms variables
        Wtot = 0.;
        Ytot = 0.;
        Ctot = 0.;
        tmp  = 0.;
        Pmin = 1.e+300 ;

        // Boltzmann average for updating unemployment share
        if(beta>0.){
            wage_norm = 0.;
            for(i=0;i<N;i++)if(ALIVE[i]==1){
                arg = beta*(W[i]-Wmax)/Wavg ;
                if(arg > -100.) wage_norm += std::exp(arg);
            }
        }
        new_len = 0 ;
        deftot = 0.;
        firm_savings = 0.;
        debt_tot = 0.;

        // Dependence of inflation to trust of firms in CB
        if(TAU_TAR_NOT_CONST){
            if(phi_pi < ZERO || pi_target < ZERO){
                tau_tar = tau_tar_init;
            }

            else{
            arg = - alpha_i*fabs(inflation-pi_target)/(phi_pi*pi_target); //old definition
            //arg = - alpha_i*fabs(inflation-pi_target)/(pi_target);
            tau_tar = (1.-omega_i)*tau_tar + omega_i*std::exp(arg);//*2*phi_pi/(1+phi_pi);
            tau_tar = std::max(std::min(1., tau_tar), 0.);
            tau_r = 1 - tau_tar;
            }
        }

        double pi_used = tau_tar * pi_target + tau_r * pi_avg ;
        double frag_avg = 0.0; // Average fragility

        alpha_g = alpha_g_0*theta0; // adjust alpha g by theta because we changes phi
        Gamma = std::max(alpha_g * (rm_avg-pi_used),Gamma0); //set Gamma


        for(i=0;i<N;i++){

            // living firms
            real_consumption[i] = std::min(D[i], Y[i]);
            if(ALIVE[i]==1){
                pay_roll = (Y[i]*W[i]/zeta) ; //Denominator for most quantities here
                total_sales = P[i]*std::min(D[i], Y[i]); 

                // interest paid per unit production by firms

                // if not bankrupt update price / production / wages and compute interests / savings and debt

                if((A[i] > -theta*total_sales)||(theta<0.)){

                    //frag_avg += (-1.*A[i]*Y[i]/pay_roll);
                    if (A[i]>ZERO){
                        frag_avg += ZERO;
                    }
                    else{
                        frag_avg += (-1.*A[i]*Y[i]/(theta*total_sales));
                    }

                    if(total_sales>0.){ // pay_roll>0.){
                        //ren = Gamma * A[i] / pay_roll;
                        if (A[i]>ZERO){
                            ren = 0; // for healty firms the gamma rule does not apply See COVID Inflation Paper
                        }
                        else{
                            ren = Gamma * A[i] / (theta*total_sales);
                        }
                        //ren = Gamma *  (-1.*A[i]*Y[i]/total_sales);
                    }  //$\Gamma \Epsilon_{i}/(W_{i} Y_{i})
                    else{
                        std::cout << t << ", Total sales <= 0: " << total_sales << ", " << D[i] << ", " << P[i] << ", " << Pavg << ", " <<  Y[i]<< ", " << A[i]<< ", " << " Firm: " << i << std::endl;
                        ren = 0.;

                    }


                    if (ren> 1.) ren=  1. ;
                    if (ren < -1.) ren= -1. ;

                    rp = gammap*dis(gen); // $\gamma_{p} \xi_{i}(t)$
                    rw = gammaw*dis(gen); // $\gamma_{w} \xi_{t}(t)$

                    dY = D[i] - Y[i] ;
                    p  = P[i];

                    if(beta>0.){
                        arg = beta*(W[i]-Wmax)/Wavg ;
                        u_share = 0.;
                        if(arg > -100.)u_share = u * N * (1.-bust) * std::exp(arg) / wage_norm;//Eqn(A9)
                    }

                    else{
                        u_share = u;
                    }

                    //excess demand
                    if(dY>0.){

                        //increase production
                        double eta = eta0p*(1.+ren); //Eqn(10) from Paper2
                        if(eta<0.0){
                            eta=0.0;
                        }// Clipping values between [0,1]
                        if(eta>1.0){
                            eta=1.0;
                        }
                        etaplus[i] = eta; // store this firms eta_{+}

                        Y[i] += std::min(eta*dY,u_share*zeta); // Eqn (9) -> Excess demand paper2

                        //increase price
                        if(p<Pavg){
                            if(Y[i]>ZERO){
                                P[i] *= (1. + (rp*D[i]/Y[i])) ; // Eqn(A8) -> top bit. from paper 1 P[i] *=  ;
                                //P[i] *= ipp; // interest paid per unit production
                            }
                        }
                        //increase wage
                        if((PROFITS[i]>0.)&&(gammaw>ZERO)){

                            // why do you have gamma_w > 0 condition
                            W[i] *= 1. + (1.0+ren) * rw * e; //Eqn(A11) top bit. Sign change because we compute the fragility factor without the negative sign.

                            W[i] = std::min(W[i], zeta*(P[i]*std::min(D[i], Y[i]) + rhol*std::min(A[i],0.) + rhod*std::max(A[i],0.0))/(Y[i]));// Set wages so that profits never go to zero even with this new wage.
                            W[i] = std::max(W[i],0.); // Wages can never be zero.

                        }
                    }

                    //excess production
                    else {

                        //decrease production
                        double eta = eta0m*(1.-ren);// Eqn(10) paper2
                        if(eta<0.0){
                            eta=0.0;// Clipping values between [0,1]
                        }
                        if(eta>1.0){
                            eta=1.0;
                        }
                        etaminus[i] = eta; // Store this firms eta_{-}

                        Y[i] += eta*dY ; // Eqn 9 Paper2

                        //decrease price
                        if(p>Pavg){
                            if(D[i]>ZERO){
                                //P[i] = (1. - (rp*((Y[i]/D[i])-1))) ;
                                if (Y[i]/D[i] > 1/rp){
                                    P[i] *= (1. - (rp*Y[i]/D[i])) ; //prevent from negative prices
                                }
                                else{
                                    P[i] *= (1. - (rp*Y[i]/D[i])) ; //Eqn 12 (lower bit) *Y[i]/D[i]
                                //P[i] *= ipp; // interest paid per unit production
                                }
                            }
                        }
                        //decrease wage
                        if(PROFITS[i]<0.){
                            W[i] *= 1. - (1.0-ren)*rw*u; // Eqn (A11) lower bit
                            W[i] = std::max(W[i],0.);
                        }
                    }

                    if(DOANTICIPATE){
                        P[i] *= 1.0 + gamma_p * pi_used; //set inflation expectations
                        W[i] *= 1.0 + wage_factor * pi_used; // set inflation expectations
                    }


                    Y[i] = std::max(Y[i],0.);

                    Wtot += W[i]*Y[i];
                    //tmp  += P[i]*Y[i];
                    tmp  += P[i]*real_consumption[i];
                    Ytot += Y[i];
                    Ctot += real_consumption[i];

                    firm_savings += std::max(A[i],0.); // Eqn (A7) paper1
                    debt_tot     -= std::min(A[i],0.); // Eqn (A7) paper 1

                    Pmin = std::min(Pmin,P[i]);

                    if((P[i]>1.0/ZERO)||(P[i]<ZERO)){
                        printf("price under/overflow... (1)\n");
                        std::cout << " Price " << P[i] << " of firm  " << i << " Output/demand: " << Y[i]/D[i] << " rp: " << rp << " Imbalance decrease: " << (1. - (rp*std::max(Y[i]/D[i], 0.8))) << "Inflation adjustment " << 1.0 + gamma_p * pi_used <<std::endl;
                        if(P[i]>1.0/ZERO)collapse=3;
                               if (P[i]<ZERO) collapse=4;
                        t=tsim;
                    }
                }

                // if bankrupt shut down and compute default costs
                else { // This company was alive in the previous time-step but has gone bankrupt now
                    deftot -= A[i]; // Eqn(A6) Paper1
                    Y[i] = 0.;
                    ALIVE[i] = 0;
                    A[i] = 0.;
                    new_list[new_len]=i;
                    new_len++;

                }
            }
            // for companies already dead
            else{
                new_list[new_len]=i;
                new_len++;
            }
        }
        
        //Pavg = tmp / Ytot ; // Average price pbar = p_{i} Y_{i} /(\sum_{i} Y_{i})
        Pavg = tmp / Ctot; // Aver price pbar = \sum_i p_i c^R_i /\sum_i c^R_i  Consumer price index
        Wavg = Wtot / Ytot ; // Average wage = W_{i} Y_{i}/(sum_{i} Y_{i})

        e = Ytot / N ;//Computing employment
        e /= zeta;

        u = 1. - e ; // Unemployment


	double ytot_temp = Ytot; 

        
        /* *********************************** INTERESTS ************************************ */
        Atot =  0;
        for(i=0;i<N;i++){
            Atot += A[i];
        }

        double left = 0.0;
        if (energy_sector<ZERO && energy_sector>-ZERO){

            energy_sector = 0.0;
        } 
        if (fabs(S + firm_savings - deftot - debt_tot - M0 + energy_sector) > 0.0){


            
            left = S + firm_savings - deftot - debt_tot - M0 + energy_sector;


            if ( fabs(left) > (S+firm_savings)*ZERO ){
                std::cout << "Huge problem" << std::endl;
                std::cout << t << "\t" << S << "\t" << firm_savings << "\t" << M0 << "\t" << deftot <<  "\t" << debt_tot << "\t" << S + firm_savings - deftot - debt_tot << "\t" << left << "\t" << energy_sector << "\t"<< Atot << std::endl ;
                 exit(1);
            }

            else{
                S -= left;
            }

            if (S < 0.0){
                std::cout << "Negative Savings" << std::endl;
                exit(1);
            }
        }


        double temp_rhol;
        double temp_rhod;

        temp_rhol = rho;
        if(debt_tot>0.)temp_rhol += (1.-f)*deftot / debt_tot ; // Change f -> 1-f in Eqn(7) Paper1

        interests = temp_rhol*debt_tot ;

        temp_rhod = k = 0.;
        if( S + firm_savings > 0. ){
            temp_rhod = (interests - deftot) / ( S + firm_savings ); // From Eqn(8) Paper1
            k = debt_tot / ( S + firm_savings) ;
        }

        rhod = temp_rhod;
        rhol = temp_rhol;

        S += rhod*S;
        /* ******************************* SHOCK HAPPENS HERE ********************************* */

        propensity = G * (1.+ alpha*(pi_used-rp_avg) ) ; // Eqn(5) paper 2
        propensity = std::max(propensity,0.); // Clipping values between [0,1]
        propensity = std::min(propensity,1.);

        if(shock_details.shock.shock_array[t] || (t > t_start && t < tprod_shock))
        {
            if(shock_details.shock.shock_array[t]){
                std::cout << t << ", Consumption shock" << std::endl;
                //propensity = propensity*cfactor;
                propensity = propensity*cfactor_array[counter];
                
                //std::cout << t << " propensity " << propensity << " cfactor " << cfactor << " " << G << " " << alpha << std::endl;
            }

            switch(shock_details.shocktype)
            {
                case production:
                case prod_debt:
                    std::cout << t << ", Production shock" << std::endl;
                    //std::cout << t << " Total Production " << Ytot << " Total Demand " << Dtot << " Cash Balance " << Atot << std::endl;
                    if(shock_details.shock.shock_array[t]){
                        //zfactor = 0.9; 
                        zfactor = zfactor_array[counter];
                        //std::cout << t << " zfactor " << zfactor << std::endl; //0.9
                    }
                    else {
                        //zfactor = zfactor_tmp;
                        zfactor = zfactor_array[counter];
                        //std::cout << t << " zfactor normal"<< zfactor << std::endl;
                        }
                    zeta = zfactor*zeta0;
                    counter = counter + 1;
                    //std::cout << t << " counter "<< counter << std::endl;
		    
		    for (int i =0; i< Y.size(); i++){
		      Y[i] *= zfactor;
		    }
		    Ytot *= zfactor;
                    //Ytot *= zfactor;
                    //std::cout << t << " Current zeta " << zeta << " Old zeta " << zeta0 << std::endl;
                    break;
                default:
                    break;
            }
        }
        else{

            for (int i = 0; i < Y.size(); i++){
                 Y[i] *= zeta0/zeta;
            }
            Ytot *= zeta0/zeta;
            zeta = zeta0;
        }

        if (t>= t_end && t <= t_end+extra_steps && extra_cons == 1){
            std::cout << "Increasing consumption" << std::endl;
            propensity = propensity+0.2;
            propensity = std::min(propensity, 1.0); 
            std::cout << t << " propensity " << propensity << " " << G << " " << alpha << std::endl;
        }

       
        if (debt_policy.shock_array[t]){
            true_end = t;
            if (adapt == 1){
                if (t <= policy_end){
                    theta = theta0; //*1000
                    //std::cout << t << " Theta =  " << theta << " prolong condition: "<< kappa*(frag_avg*theta/Ytot)<<" fragavg: "<<  frag_avg/Ytot << std::endl;
                    debt_policy.shock_array[t+1] = true;
                    if (kappa*(frag_avg*theta/Ytot) > theta0 ){
                        theta = std::max(theta0, kappa*frag_avg*theta/Ytot);
                    }
                    std::cout << t << " Theta =  " << theta << std::endl;
                }
                else if (kappa*(frag_avg*theta/Ytot) > theta0 ){
                    //std::cout << t << "Increase adaptive policy one time step, Theta = " << theta << " condition " << kappa*frag_avg*theta/Ytot << " fragavg: " << frag_avg/Ytot<< std::endl;
		            theta = std::max(theta0, kappa*frag_avg*theta/Ytot);
                    //std::cout << t << "New Theta = " << theta << std::endl;
                    debt_policy.shock_array[t+1] = true;
                }
                else{
                    //std::cout << t << ", Condition not fulfilled, Theta = " << theta << " condition " << kappa*frag_avg*theta/Ytot << " fragavg: "<< frag_avg/Ytot << std::endl;
                    theta = theta0;
                    
                }
            }
            else{
                std::cout << "Naive policy being applied" << std::endl;
                theta = theta0*100;
                std::cout << t << "Theta = " << theta << std::endl;
            }
        }

        else{
            theta = theta0;
	    Gamma0 = Gammatemp; 
        }

        gamma_e = gamma_e_0;
        if (t==price_start-1 || t==price_end + 1 ) {std::cout << t << ", " << Pavg << " Pavg after "  << "price shock" << gamma_e << ", " <<pfactor_array[counter_p] << "  p_e, " << p_e <<std::endl;}
        if (price_shock.shock_array[t]){
            std::cout << t << ", Price shock" << std::endl;
            std::cout << Pavg << " Pavg before" << std::endl;
            for (int i = 0; i < P.size(); i++){

                 tmp_p = P[i];

                 P[i] *= (1+ gamma_e *pfactor_array[counter_p]); //std::max(0.0, pfactor_array[counter_p])

            }
            p_e += pfactor_array[counter_p];

            Pmin *= (1+ gamma_e * pfactor_array[counter_p]);
            Pavg *= (1+ gamma_e * pfactor_array[counter_p]);
            std::cout << t << ", " << Pavg << " Pavg after "  << "price shock" << gamma_e << ", " <<pfactor_array[counter_p] << "  p_e, " << p_e <<std::endl;
            counter_p = counter_p + 1;


        }

    // Can be deleted, was only to manually include helicopter money
    if (t >= price_end+1 && t <= price_end+1){
            std::cout << t << " Perform Helicopter drop" << std::endl;
            std::cout << "Increasing Savings of Households" << std::endl;
            double S0 = S;
            S += kappa_h*S0; //0.2*S0
            M0 +=  kappa_h*S0;//0.2*S0
        }
    // Delete until here 

    if (t ==t_end){
        if (helico ==1 & shockflag > 0){
            std::cout << t << " Perform Helicopter drop" << std::endl;
            std::cout << "Increasing Savings of Households" << std::endl;
            double S0 = S;
            S += 0.5*S0;
            M0 +=  0.5*S0;
        }
    }

    // Wind fall tax
    if (t >= price_end-12 && t <= price_end+12){
        delta_e=delta_w;
        std::cout << t << " Windfall tax" << delta_e <<std::endl;}
    else{ delta_e = delta_e_0;}


    //    ytot_temp = Ytot; 
     /* *********************************** CONSUMPTION ************************************ */

        budget = propensity * ( Wtot/zeta + std::max(S,0.) ); // Eqn (5) Paper 2

        Pnorm = 0.; // Boltzmann average for price. Used in eqn (A3) paper 1
        for(i=0;i<N;i++)if(ALIVE[i]==1){
            arg = beta*(Pmin-P[i]) / Pavg ;
            if(arg > -100.) Pnorm += std::exp(arg);
        }

        Dtot = 0.;
        profits = 0.;
        firm_savings = 0.;


        for(i=0;i<N;i++)if(ALIVE[i]==1){

            D[i] = 0.;

            arg = beta*(Pmin-P[i])/Pavg ;

            if(arg > -100.) D[i] = budget * std::exp(arg) / Pnorm / P[i]; //Eqn (A3) Paper 1


            PROFITS[i]  = P[i]*std::min(Y[i],D[i]) - (Y[i]*W[i]/zeta) + rhol*std::min(A[i],0.) + rhod*std::max(A[i], 0.) - gamma_e*p_e*Y[i];
            energy_sector += gamma_e*p_e*Y[i];

            // Eqn (A12) Paper 1
            S          -= P[i]*std::min(Y[i],D[i]) - (Y[i]*W[i]/zeta); // Eqn (A14) without dividend term
            A[i]       += PROFITS[i]; // Eqn (A13) without dividend term


            // Retain sum 0 Money

            //Dividend payments
            if((A[i]>0.)&&(PROFITS[i]>0.)){
                //Dividends paid as a fraction of the profits
                if(dividends=='P'){
                    S    += delta*PROFITS[i];
                    A[i] -= delta*PROFITS[i];
                }
                //Dividends paid as a fraction of the cash balance
                if(dividends=='A'){
                    S    += delta*A[i];
                    A[i] -= delta*A[i];
                }
            }

            Dtot    += D[i];
            profits += PROFITS[i];

            firm_savings += std::max(A[i],0.);


            }


        if (t<1000) delta_e = 0.2;
        if (t>1000 && t<2000) delta_e = 0.1;
        if (t>2000 && t<3000) delta_e = 0.05;
        if (t>3000 && t<3500) delta_e = delta_e_0;
        if(dividends=='A'){
            S    += delta_e*energy_sector;
            energy_sector -= delta_e*energy_sector;
        }

        // Beyond this point there is only the revival bit and nothing else
        // So we define the u-rate and bust here
        //
        double u_bef = u;
        double bust_bef = (N - sum_vector(ALIVE))/N;

        /* ******************************* REVIVAL ******************************** */

        //revival
        deftot = 0.;
        for(i=0;i<new_len;i++)if(dis(gen)<phi){

            new_firm = new_list[i];
            Y[new_firm] = std::max(u,0.)*dis(gen);
            etaplus[i]  = eta0p;
            etaminus[i] = eta0m; 
            ALIVE[new_firm] = 1;
            if (u_bef == 1 ){
                P[new_firm] = 1.0 + (1e-6 * bernouilli_trial(gen));
                W[new_firm] = 1.0;
                A[new_firm] = 0.0;
                D[new_firm] = Y[new_firm] * (1 + 1e-6 * bernouilli_trial(gen));
            }
            else{
                P[new_firm] = Pavg;
                W[new_firm] = Wavg;
                A[new_firm] = (W[new_firm]*Y[new_firm]);
                D[new_firm] = Y[new_firm] * (1 + 1e-6 * bernouilli_trial(gen));
            }

            deftot  += A[new_firm];

            firm_savings += A[new_firm];

            PROFITS[new_firm] = 0.;
            }




        /* ******************************* FINAL ******************************** */

        //new averages
        tmp  = 0.;
        Ytot = 0.;
        Ctot = 0.;
        Wtot = 0.;
        bust = 0.;
        Wmax = 0.;
        Atot = 0.;
        double etaplus_avg = 0.0;
        double etaminus_avg = 0.0;
        debt_tot = 0.;
        int firms_alive = 0;


        for(i=0;i<N;i++){

            //final averages
            if(ALIVE[i]==1){

                if((firm_savings>0.)&&(A[i]>0.))A[i] -= deftot*A[i]/firm_savings;

                Wtot    += Y[i]*W[i];
                Ytot    += Y[i];
                Ctot    += real_consumption[i];
                //tmp     += P[i]*Y[i];
                tmp     += P[i]*real_consumption[i];

                Wmax = std::max(W[i],Wmax);

                debt_tot -= std::min(A[i],0.);
                Atot     += A[i];

                etaplus_avg += etaplus[i];
                etaminus_avg += etaminus[i];
                firms_alive +=1;

                HHindex += (real_consumption[i]*real_consumption[i]);

            }

            else bust += 1./N;

        }

        //Pavg = tmp / Ytot;
        Pavg = tmp / Ctot;
        Wavg = Wtot / Ytot;
        if (t==price_end) {p_e = Pavg;}
        inflation = (Pavg-Pold)/Pold;
        Pold = Pavg;
        HHindex = HHindex / (Ctot*Ctot);

        p_e *= (1+ inflation);

        e = Ytot / N  ;
        //std::cout << t << ", End Pavg: " << Pavg << std::endl;
        //e /=zeta; 
        if((e-1 > ZERO)|| (e < -ZERO) || (S < 0.0) ){
            printf("Error!! -> t =%d\t e = %.10e\tS=%.10e\n",t+1,e,S);
            std::cout << "Zeta here " << zeta << " Propensity " << propensity << " Atot " << Atot << " firm savings " << firm_savings << " Dtot " << Dtot << " Ytot " << Ytot  << std::endl;
            collapse=2;
            t=tsim;
        }
        e /= zeta;
        e = std::min(e,1.);
        e = std::max(e,0.);

        u =  1. - e;

        if(Ytot<ZERO){
            printf("Collapse\n");
            collapse = 1;
            t=tsim;
        }

        /************************* INTEREST RATE SETTING ************************************/


        // Dynamic CB policy

        
        rho = rho0 + phi_pi * ( pi_avg - pi_target - pi_p);



        /************************************** OUTPUT ************************************/


        if((t%tprint==0)){
            double R_temp = eta0p/eta0m;
            // Implementing the output allows for easily adding other variables to
            // be saved into output vector.
            output_vector = Vector{static_cast<double>(t-Teq), u_bef, bust_bef, Pavg, Wavg, S, Atot, firm_savings, debt_tot, inflation, pi_avg, budget, k, Dtot, rhol, rho, rhod, pi_used, tau_tar, tau_r, energy_sector}; //R_temp

            output_vector.push_back(Wtot);
            output_vector.push_back(etaplus_avg/firms_alive);
            output_vector.push_back(etaminus_avg/firms_alive);
            output_vector.push_back(w_nominal);
            output_vector.push_back(Ytot);
            output_vector.push_back(deftot);
            output_vector.push_back(profits);
            output_vector.push_back(debt_tot/S);
            output_vector.push_back(static_cast<double>(firms_alive));
            output_vector.push_back(left);
            output_vector.push_back(u);
            output_vector.push_back(bust);
            output_vector.push_back(frag_avg/Ytot);
            output_vector.push_back(true_end);
            output_vector.push_back(theta);
	    output_vector.push_back(ytot_temp);
	    output_vector.push_back(std::min(Ytot, ytot_temp)); 


            output.write_vector_to_file(output_vector);

            output_vector.clear();
        }

    }


    output.close_file(); 

    return 0; 
}
