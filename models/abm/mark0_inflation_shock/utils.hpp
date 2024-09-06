#ifndef UTILS_H
#define UTILS_H
#pragma once

#include "cxxopts.hpp"
#include <fstream>
#include <cassert>
#include <stdexcept>
typedef std::string String;
typedef std::vector<double> Vector;
typedef std::vector<String> VectorString;
typedef std::vector<double * > VectorPtr;
typedef cxxopts::ParseResult PR;

class Parameters{
  public:
    VectorString param_names_double;
    VectorString param_names_int;
    String outputname;
    String projectname;
    String projectdesc;

    cxxopts::Options *options;

    Parameters (String s1, String s2):
      projectname(s1), projectdesc(s2),
      param_names_double({"R0", "theta", "Gamma0", "rho0", "alpha", "phi_pi", "alpha_e", "pi_star", "e_star", "tau_tar", "wage_factor","y0",  "gammap", "eta0m", "tau_r", "alpha_g","cfactor", "zeta", "zfactor", "kappa", "G0", "phi", "omega", "delta"}),
      param_names_int({"seed", "shockflag", "t_start", "t_end", "policy_start", "policy_end", "helico", "nfirms", "extra_cons", "adapt", "extra_steps", "tsim", "teq", "tprint", "renorm", "cbon"}){

      options = new cxxopts::Options(projectname, projectdesc);

      options->add_options()
        ("R0", "Ratio of hiring-firing rate", cxxopts::value<double>()->default_value("2.0"))
        ("theta", "Maximum credit supply available to firms", cxxopts::value<double>()->default_value("3.0"))
        ("Gamma0", "Financial Fragility sensitivity", cxxopts::value<double>()->default_value("0.0"))
        ("rho0", "Baseline interest rate", cxxopts::value<double>()->default_value("0.0"))
        ("alpha", "Influence of deposit rates on consumption", cxxopts::value<double>()->default_value("0.0"))
        ("phi_pi", "Intensity of interest rate policy of central bank", cxxopts::value<double>()->default_value("0.0"))
        ("alpha_e", "Influence of employment on interest rate policy of central bank", cxxopts::value<double>()->default_value("0.0"))
        ("alpha_i", "Influence of inflation of firm expectations", cxxopts::value<double>()->default_value("1.0"))
        ("pi_star", "Inflation Target", cxxopts::value<double>()->default_value("0.0"))
        ("e_star", "Employment Target", cxxopts::value<double>()->default_value("0.0"))
        ("tau_tar", "Inflation target parameter", cxxopts::value<double>()->default_value("0.0"))
        ("wage_factor", "Factor to adjust wages to inflation expectations", cxxopts::value<double>()->default_value("1.0"))
        ("y0", "Initial production", cxxopts::value<double>()->default_value("0.5"))
        ("gammap", "Parameter to set adjustment of prices", cxxopts::value<double>()->default_value("0.01"))
        ("eta0m", "Firing propensity", cxxopts::value<double>()->default_value("0.2"))
        ("tau_r", "Realised inflation parameter", cxxopts::value<double>()->default_value("0.0"))
        ("alpha_g", "Influence of loans interest rate on hiring-firing policy", cxxopts::value<double>()->default_value("0.0"))
        ("seed", "seed for random number generation", cxxopts::value<double>()->default_value("1"))
        ("shockflag", "Flag to set kind of shock", cxxopts::value<double>()->default_value("0"))
        ("t_start", "Time when shock occurs", cxxopts::value<double>()->default_value("2000"))
        ("t_end", "Time when shock end", cxxopts::value<double>()->default_value("2005"))
        ("policy_start", "Time when debt policy starts", cxxopts::value<double>()->default_value("2000"))
        ("policy_end", "Time when debt policy ends", cxxopts::value<double>()->default_value("2005"))
        ("output", "User specified output file name. If not specified program generates the name based on other parameters", cxxopts::value<std::string>()->default_value("out"))
        ("cfactor", "Factor by which to reduce consumption during shock", cxxopts::value<double>()->default_value("0.5"))
        ("zeta",
       "Labor productivity factor", cxxopts::value<double>()->default_value("1.0"))
        ("zfactor", "Factor by which to reduce production during shock", cxxopts::value<double>()->default_value("0.5"))
        ("kappa","Offset for adaptive policy", cxxopts::value<double>()->default_value("1.25"))
        ("helico", "Do helicopter money drop", cxxopts::value<double>()->default_value("0.0"))
        ("nfirms", "Number of firms in the economy", cxxopts::value<double>()->default_value("10000") )
        ("extra_cons", "Agents consume more after shock", cxxopts::value<double>()->default_value("0.0"))
        ("adapt", "Have adaptive policy", cxxopts::value<double> ()->default_value("0.0"))
        ("extra_steps", "Number of time steps during which consumption is higher than baseline", cxxopts::value<double>()->default_value("1.0"))
        ("tsim", "Simulation length", cxxopts::value<double>()->default_value("7000"))
        ("teq", "Equilibration time", cxxopts::value<double>()->default_value("200"))
        ("tprint", "Frequency to write to output file", cxxopts::value<double>()->default_value("1"))
        ("G0", "Fraction of savings to consume", cxxopts::value<double>()->default_value("0.5"))
        ("phi", "Revival probability per unit time", cxxopts::value<double>()->default_value("0.1"))
        ("omega", "Parameter for moving average", cxxopts::value<double>()->default_value("0.2"))
        ("omega_i", "Parameter for moving average for inflation expectation", cxxopts::value<double>()->default_value("0.2"))
        ("delta", "Fraction of dividends to be distributed", cxxopts::value<double>()->default_value("0.02"))
        ("gamma_e", "Factor of price increase", cxxopts::value<double>()->default_value("1"))
        ("delta_p", "Price change", cxxopts::value<double>()->default_value("0.2"))
        ("renorm", "Renormalize in units of the price ", cxxopts::value<double>()->default_value("1"))
        ("gamma_cb", "Increased activity of CB", cxxopts::value<double>()->default_value("5.0"))
        ("theta_cb", "Threshold when CB gets active ", cxxopts::value<double>()->default_value("0.002"))
	("cbon", "Switch on CB at end of shock", cxxopts::value<double>()->default_value("0"))
  ("tprod", "Additional time of productivity shock", cxxopts::value<double>()->default_value("0"))
  ("price_start", "Start of price shock", cxxopts::value<double>()->default_value("2000"))
  ("price_end", "End of price shock", cxxopts::value<double>()->default_value("2000"))
  ("kappa_h", "Strength of helicopter money", cxxopts::value<double>()->default_value("0.2"))
  ("delta_w", "Dividends for Windfall tax", cxxopts::value<double>()->default_value("0.08"))
  ("delta_e", "Dividends of Energy Sector", cxxopts::value<double>()->default_value("0.04"))
  ("shock_st", "Strength of shock", cxxopts::value<double>()->default_value("1"))
  ("gamma_p", "Price factor", cxxopts::value<double>()->default_value("1"))
        ("h,help", "Print usage");
      }


    void parse_cmdline(int argc, char** argv, VectorPtr &ptr_vec, std::vector< int * > &ptr_vec_int ){


      PR cmdline_args = this->options->parse(argc, argv);

      if (cmdline_args.count("help")){
        std::cout << this->options->help() << std::endl;
        exit(0);
      }


      assert(this->param_names_double.size() == ptr_vec.size());

      for (int i = 0; i < ptr_vec.size(); i++){
        *ptr_vec[i] = cmdline_args[this->param_names_double[i]].as<double>();
      }


      assert(this->param_names_int.size() == ptr_vec_int.size());
    

      for (int i =0; i < ptr_vec_int.size(); i++){
        // This bit is a little hack to be able to call the program from python
        // The issue is that python when converting to string adds decimal points to ints
        // causing trouble to the parser. So we simply cast the input from the command line to int.
        double parsed_input = cmdline_args[this->param_names_int[i]].as<double>();
        *ptr_vec_int[i] = static_cast<int>(parsed_input);
      }

      // If user passes a specific outputfilename then parse that too

      if (cmdline_args.count("output")){
        this->outputname = cmdline_args["output"].as<std::string>();
      }
  }
};



enum ShockType{
      no_shock, // no shocks
      consumption, // only consumption
      production, //production shock
      cons_and_debt, // consumption shock + debt relief
      prod_debt, // Production + debt
};

const int nb_shocktypes = 5;
String map_enum_str(ShockType const &shock_type){

  typedef std::pair<int,String> string_map;
  auto m = [](ShockType const& s,String const& str){return string_map(static_cast<int>(s),str);};
  std::vector<string_map> const my_map=
    {
        m(no_shock,"base"), 
        m(consumption, "cons_pure"),
        m(production, "cons_prod"),
        m(cons_and_debt, "cons_theta"),
        m(prod_debt, "prod_debt")
  };

  for(auto i  : my_map){
    if(i.first==static_cast<int>(shock_type))
    {
      return i.second;
    }
  }
  return "";
}

 

struct Shocks{
  int t_start; // start of shock
  int t_end; //end of shock
  int t_simulation; //simulation time
  std::vector<bool> shock_array;
  Shocks () {}
  Shocks(int a, int b, int c): t_start(a), t_end(b), t_simulation(c){

    //Populate the bool array
    for (int t = 0; t<this->t_simulation; t++){
      shock_array.push_back(this->is_shock(t, this->t_start, this->t_end));
    }
  }


  bool is_shock(int t, int t_start, int t_end){
    return (t >= t_start) && (t < t_end);
  }

  void no_shock(){
    std::fill(this->shock_array.begin(), this->shock_array.end(), false);
  }
};

struct ShockDetails{
  int shockflag;
  int t_start, t_end, t_simulation;
  Shocks shock;
  ShockType shocktype;
  String filename;
  ShockDetails (){}
  ShockDetails(int a, int b, int c, int d){

    if (a >= nb_shocktypes || a < 0){
      throw std::invalid_argument("Unrecognised flag.");
    }
    this->shockflag = a;
    this->t_start = b;
    this->t_end = c;
    this->t_simulation = d;
    shock = Shocks(t_start, t_end, t_simulation);
    shocktype = ShockType(shockflag);
    filename = map_enum_str(shocktype);

    if (this->shockflag == 0){
      shock.no_shock();
    }
  }
};

class FileWriter{


public:
  std::ofstream file_output;
  String filename;

  FileWriter(String f) : filename(f) {
    this->file_output.open(filename);
  }

  FileWriter() {}


  void close_file(){
    this->file_output.close();
  }

  template <typename T>
  void write_vector_to_file(std::vector<T> vector_to_write){
    for (typename std::vector<T>::iterator it = vector_to_write.begin(); it != vector_to_write.end()-1; it++){
      this->file_output << *it << "\t";
    }
    if (!vector_to_write.empty()){
      this->file_output << vector_to_write.back() << std::endl;
    }
  }

  template <typename T>
  void write(T data){
    assert(this->file_output.is_open());
    this->file_output << data << std::endl;
  } 
};

  
#endif /* UTILS_H */
