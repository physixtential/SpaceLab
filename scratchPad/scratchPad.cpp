#include <chrono>
// 'compressed' chrono access --------------vvvvvvv
typedef std::chrono::high_resolution_clock  HRClk_t; // std-chrono-hi-res-clk
typedef HRClk_t::time_point                 Time_t;  // std-chrono-hi-res-clk-time-point
typedef std::chrono::milliseconds           MS_t;    // std-chrono-milliseconds
typedef std::chrono::microseconds           US_t;    // std-chrono-microseconds
typedef std::chrono::nanoseconds            NS_t;    // std-chrono-nanoseconds
using   namespace std::chrono_literals;  // support suffixes like 100ms, 2s, 30us
#include <iostream>
#include <iomanip>
#include <thread>


class Ansi_t   // use ansi features of gnome-terminal, 
{              // or any ansi terminal
   // note: Ubuntu 15.10 gnome-terminal ansi term cursor locations 
   //       are 1-based, with origin 1,1 at top left corner

    enum ANSI : int { ESC = 27 }; // escape 

public:

    static std::string clrscr(void)
    {
        std::stringstream ss;
        ss << static_cast<char>(ESC) << "[H"   // home
            << static_cast<char>(ESC) << "[2J"; // clrbos
        return(ss.str());
    }

    //       r/c are 0 based------v------v------0 based from C++
    static std::string gotoRC(int r, int c)
    {
        std::stringstream ss;
        // Note: row/col of my apps are 0 based (as in C++)
        // here I fix xy to 1 based, by adding 1 while forming output
        ss << static_cast<char>(ESC) << "["
            << (r + 1) << ';' << (c + 1) << 'H';
        return(ss.str());
    }

    // tbr - add more ansi functions when needed

}; // Ansi_t

int main(int, char**)
{
    int retVal = -1;
    {
        Time_t start_us = HRClk_t::now();

        {
            std::cout << Ansi_t::clrscr() << std::flush;  // leaves cursor at top left of screen

            for (int i = 0; i < 10; ++i)
            {

                for (int r = 0; r < 5; ++r) // 0 based
                {
                    std::cout << Ansi_t::gotoRC(r + 5, 5)  // set cursor location
                        << "-----" << std::flush;
                }

                std::this_thread::sleep_for(500ms);

                // to overwrite
                for (int r = 0; r < 5; ++r)
                {
                    std::cout << Ansi_t::gotoRC(r + 5, 5)  // set cursor location
                        << "xxxxx" << std::flush;
                }

                std::this_thread::sleep_for(500ms);

                std::cout << Ansi_t::gotoRC(11, 5) << 9 - i << std::flush;

            }// for i


            std::cout << "\n\n" << std::endl;

            return 0;
        }

        auto  duration_us = std::chrono::duration_cast<US_t>(HRClk_t::now() - start_us);

        std::cout << "\n\n  duration  " << duration_us.count() << " us" << std::endl;
    }

    return(retVal);
}