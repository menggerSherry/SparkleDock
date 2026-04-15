#pragma once
#include <iostream>
#include <vector>
#include <string> 



struct FastDifire
{
    // res name and its number
    map<string,int> RES_3 = {
        {"ALA", 0}, 
        {"CYS", 1}, 
        {"ASP", 2}, 
        {"GLU", 3}, 
        {"PHE", 4}, 
        {"GLY", 5}, 
        {"HIS", 6}, 
        {"ILE", 7}, 
        {"LYS", 8}, 
        {"LEU", 9}, 
        {"MET", 10}, 
        {"ASN", 11}, 
        {"PRO", 12}, 
        {"GLN", 13}, 
        {"ARG", 14}, 
        {"SER", 15}, 
        {"THR", 16}, 
        {"VAL", 17}, 
        {"TRP", 18}, 
        {"TYR", 19}, 
        {"MMB", 20}
    };

    map<string, int> atomnumber = {
        {"ALAN", 0}, {"ALACA", 1}, {"ALAC", 2}, {"ALAO", 3}, {"ALACB", 4},
        {"CYSN", 0}, {"CYSCA", 1}, {"CYSC", 2}, {"CYSO", 3}, {"CYSCB", 4}, {"CYSSG", 5},
        {"ASPN", 0}, {"ASPCA", 1}, {"ASPC", 2}, {"ASPO", 3}, {"ASPCB", 4}, {"ASPCG", 5},
        {"ASPOD1", 6}, {"ASPOD2", 7}, {"GLUN", 0}, {"GLUCA", 1}, {"GLUC", 2}, {"GLUO", 3},
        {"GLUCB", 4}, {"GLUCG", 5}, {"GLUCD", 6}, {"GLUOE1", 7}, {"GLUOE2", 8}, {"PHEN", 0},
        {"PHECA", 1}, {"PHEC", 2}, {"PHEO", 3}, {"PHECB", 4}, {"PHECG", 5}, {"PHECD1", 6},
        {"PHECD2", 7}, {"PHECE1", 8}, {"PHECE2", 9}, {"PHECZ", 10}, {"GLYN", 0}, {"GLYCA", 1},
        {"GLYC", 2}, {"GLYO", 3}, {"HISN", 0}, {"HISCA", 1}, {"HISC", 2}, {"HISO", 3},
        {"HISCB", 4}, {"HISCG", 5}, {"HISND1", 6}, {"HISCD2", 7}, {"HISCE1", 8}, {"HISNE2", 9},
        {"ILEN", 0}, {"ILECA", 1}, {"ILEC", 2}, {"ILEO", 3}, {"ILECB", 4}, {"ILECG1", 5},
        {"ILECG2", 6}, {"ILECD1", 7}, {"LYSN", 0}, {"LYSCA", 1}, {"LYSC", 2}, {"LYSO", 3},
        {"LYSCB", 4}, {"LYSCG", 5}, {"LYSCD", 6}, {"LYSCE", 7}, {"LYSNZ", 8}, {"LEUN", 0},
        {"LEUCA", 1}, {"LEUC", 2}, {"LEUO", 3}, {"LEUCB", 4}, {"LEUCG", 5}, {"LEUCD1", 6},
        {"LEUCD2", 7}, {"METN", 0}, {"METCA", 1}, {"METC", 2}, {"METO", 3}, {"METCB", 4},
        {"METCG", 5}, {"METSD", 6}, {"METCE", 7}, {"ASNN", 0}, {"ASNCA", 1}, {"ASNC", 2},
        {"ASNO", 3}, {"ASNCB", 4}, {"ASNCG", 5}, {"ASNOD1", 6}, {"ASNND2", 7}, {"PRON", 0},
        {"PROCA", 1}, {"PROC", 2}, {"PROO", 3}, {"PROCB", 4}, {"PROCG", 5}, {"PROCD", 6},
        {"GLNN", 0}, {"GLNCA", 1}, {"GLNC", 2}, {"GLNO", 3}, {"GLNCB", 4}, {"GLNCG", 5},
        {"GLNCD", 6}, {"GLNOE1", 7}, {"GLNNE2", 8}, {"ARGN", 0}, {"ARGCA", 1}, {"ARGC", 2},
        {"ARGO", 3}, {"ARGCB", 4}, {"ARGCG", 5}, {"ARGCD", 6}, {"ARGNE", 7}, {"ARGCZ", 8},
        {"ARGNH1", 9}, {"ARGNH2", 10}, {"SERN", 0}, {"SERCA", 1}, {"SERC", 2}, {"SERO", 3},
        {"SERCB", 4}, {"SEROG", 5}, {"THRN", 0}, {"THRCA", 1}, {"THRC", 2}, {"THRO", 3},
        {"THRCB", 4}, {"THROG1", 5}, {"THRCG2", 6}, {"VALN", 0}, {"VALCA", 1}, {"VALC", 2},
        {"VALO", 3}, {"VALCB", 4}, {"VALCG1", 5}, {"VALCG2", 6}, {"TRPN", 0}, {"TRPCA", 1},
        {"TRPC", 2}, {"TRPO", 3}, {"TRPCB", 4}, {"TRPCG", 5}, {"TRPCD1", 6}, {"TRPCD2", 7},
        {"TRPCE2", 8}, {"TRPNE1", 9}, {"TRPCE3", 10}, {"TRPCZ3", 11}, {"TRPCH2", 12},
        {"TRPCZ2", 13}, {"TYRN", 0}, {"TYRCA", 1}, {"TYRC", 2}, {"TYRO", 3}, {"TYRCB", 4},
        {"TYRCG", 5}, {"TYRCD1", 6}, {"TYRCD2", 7}, {"TYRCE1", 8}, {"TYRCE2", 9}, {"TYRCZ", 10},
        {"TYROH", 11}, {"MMBBJ", 0}
    };

    
    int atom_res_trans[21 * 14] = {
        74,  75,  76,  77,  78,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   1,   2,   3,   4,   5,   0,   0,   0,   0,   0,   0,   0,   0,
        122, 123, 124, 125, 126, 127, 128, 129,   0,   0,   0,   0,   0,   0, 
        113, 114, 115, 116, 117, 118, 119, 120, 121,   0,   0,   0,   0,   0, 
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,   0,   0,   0, 
        79,  80,  81,  82,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
        130, 131, 132, 133, 134, 135, 136, 137, 138, 139,   0,   0,   0,   0, 
        25,  26,  27,  28,  29,  30,  31,  32,   0,   0,   0,   0,   0,   0, 
        151, 152, 153, 154, 155, 156, 157, 158, 159,   0,   0,   0,   0,   0,
        33,  34,  35,  36,  37,  38,  39,  40,   0,   0,   0,   0,   0,   0,
        6,   7,   8,   9,  10,  11,  12,  13,   0,   0,   0,   0,   0,   0,
        105, 106, 107, 108, 109, 110, 111, 112,   0,   0,   0,   0,   0,   0, 
        160, 161, 162, 163, 164, 165, 166,   0,   0,   0,   0,   0,   0,   0, 
        96,  97,  98,  99, 100, 101, 102, 103, 104,   0,   0,   0,   0,   0,
        140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,   0,   0,   0, 
        90,  91,  92,  93,  94,  95,   0,   0,   0,   0,   0,   0,   0,   0, 
        83,  84,  85,  86,  87,  88,  89,   0,   0,   0,   0,   0,   0,   0,
        41,  42,  43,  44,  45,  46,  47,   0,   0,   0,   0,   0,   0,   0,
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
        62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,   0,   0,
        167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    };
    // double difire_enerfy[168*168*20];
    double *difire_energy;

};

