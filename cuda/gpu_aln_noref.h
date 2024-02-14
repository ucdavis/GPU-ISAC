
/******************************************************************************

GPU based multireference alignment

Author (C) 2019, Fabian Schoenfeld (fabian.schoenfeld@mpi-dortmund.mpg.de)
Copyright (C) 2019, Max Planck Institute of Molecular Physiology, Dortmund

   This program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the Free 
Software Foundation, either version 3 of the License, or (at your option) any
later version.

   This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along with
this program.  If not, please visit: http://www.gnu.org/licenses/

******************************************************************************/

#ifndef GPU_ALN_NOREF
#define GPU_ALN_NOREF

//===================================================================[ import ]

// C core
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// C++ core
#include <cmath>
#include <ctime>
#include <array>
#include <vector>
#include <limits>
#include <iostream>

// CUDA
#include <cufft.h>
#include <cublas_v2.h>

// gpu alignment common functions
#include "gpu_aln_common.h"

//===================================================================[ common ]

extern "C" void print_gpu_info( const unsigned int device_idx );

//========================================================[ gpu alignment API ]

void gpu_alignment_init( const AlignConfig* aln_cfg, const unsigned int cuda_device_id );

extern "C" void gpu_clear();

//------------------------------------------------[ multi-reference alignment ]

extern "C" AlignParam* multi_ref_align_init(
    const AlignConfig* aln_cfg,
    const unsigned int cuda_device_id
);

extern "C" void multi_ref_align(
    const float**      sbj_data,
    const float**      ref_data,
    float*             isac_peak_list,
    float*             isac_d,
    const unsigned int isac_d_pitch,
    const float        cuda_mem_request
);

//------------------------------------------------------------[ pre-alignment ]

extern "C" AlignParam* pre_align_init(
    const unsigned int num_particles,
    const AlignConfig* aln_cfg,
    const unsigned int cuda_device_id
);

extern "C" bool pre_align_size_check(
    const unsigned int num_particles,
    const AlignConfig* cfg, 
    const unsigned int cuda_device_id, 
    const float        request, 
    const bool         verbose
);

extern "C" void pre_align_fetch(
    const float**      img_data,
    const unsigned int img_num,
    const char*        batch_type
);

extern "C" void pre_align_run( const int start_idx, const int stop_idx );

//-------------------------------------------------[ reference-free alignment ]

extern "C" AlignParam* ref_free_alignment_2D_init( 
    const AlignConfig* aln_cfg, 
    const float**      sbj_data_list,
    const float**      ref_data_list,
    const int*         sbj_cid_list,
    const unsigned int cuda_device_id
);

extern "C" bool ref_free_alignment_2D_size_check(
    const AlignConfig* cfg, 
    const unsigned int cuda_device_id, 
    const float        request, 
    const bool         verbose
);

extern "C" void ref_free_alignment_2D();

extern "C" void ref_free_alignment_2D_filter_references( const float cutoff_freq, const float falloff );

//======================================================[ CcfResultTable class]

class CcfResultTable{

    private:
        // image parameters
        unsigned int sbj_num;       // number of images to be aligned
        unsigned int ref_num;       // number of reference images
        unsigned int ring_num;      // // number of rings for polar conversion
        unsigned int ring_len;      // ring length for polar conversion
        // shift param
        unsigned int shift_num;     // number of applied shifts in total
        // ccf result table
        float* u_ccf_batch_table;   // table for subject/reference cross correlation results
        // ccf max table
        int* u_max_idx;             // pos[i] holds index position for max(row[i])
        // cuda handles
        cufftHandle cufft_pln;      // CUDA plan for the IFFT

        void  compute_max_indices();
        double interpolate_angle( 
            const unsigned int sbj_idx,
            const unsigned int max_idx,
            const unsigned int max_idx_off );

    public:
        CcfResultTable( const AlignConfig* batch_cfg );
        CcfResultTable();
       ~CcfResultTable();

        inline unsigned int row_off() const;
        inline unsigned int ref_off() const;
        inline unsigned int shift_off() const;
        inline unsigned int mirror_off() const;

        float* row_ptr( const unsigned int shift_idx, const unsigned int ref_slot_idx=0 ) const;

        inline unsigned int get_ring_num() const;
        inline unsigned int get_ring_len() const;

        inline unsigned int row_num() const;
        inline unsigned int entry_num() const;
        inline size_t memsize() const;  // batch table size in bytes
        
        void apply_IFFT();

        void compute_alignment_param(
            const unsigned int            param_idx,
            const unsigned int            param_limit,
            const vector<array<float,2>>* shifts,
            AlignParam*                   aln_param
        );
};

//=======================================================[ BatchHandler class ]

class BatchHandler{

    private:
        const char* batch_type;            // determines handler capabilities (note: this should be done by subclasses!)
        // image parameters
        unsigned int img_num;              // number of subject/reference images
        unsigned int img_dim_x;            // image dimension x-axis
        unsigned int img_dim_y;            // image dimension y-axis
        unsigned int ring_num;             // number of rings for polar conversion
        unsigned int ring_len;             // ring length for polar conversion
        // table to handle ccf results
        CcfResultTable* ccf_table;         // interface to compute and access ccf results
        // cuda handles
        cufftHandle cufft_pln;             // CUDA plan for the FFT of polar'd images in d_img_data
        cufftHandle cufft_pln_filter_in;   // CUDA plan for the FFT of   raw   images in u_img_tex_data[i]
        cufftHandle cufft_pln_filter_out;  // CUDA plan for the FFT of   raw   images in u_img_tex_data[i]
        // image buffer (processed data)
        float* d_img_data;                 // holds processed image data (polar conversion & transformed images)
        // polar conversion norm. buffer
        float* d_norm_values;
        // texture buffers (raw data)
        unsigned int img_tex_num;          // number of used texture objects
        unsigned int img_num_per_tex;      // max number of images held per texture object
        cudaTextureObject_t* img_tex_obj;  // array of texture objects
        size_t  img_tex_pitch;             // pitch for accessing texture memory allocated by cudaMallocPitch
        float** u_img_tex_data;            // array of device pointers to the data managed by the individual texture objects
                                           // NOTE: pointer array in unified memory; pointers point to device memory

        void create_texture_objects( const unsigned int num, const unsigned int tex_obj_idx );

    public:
        // constructors
        BatchHandler( const AlignConfig* cfg, const char* batch_type );
        BatchHandler();
       ~BatchHandler();

        // data acquisition
        void fetch_data( 
            const float**      img_data,
            const unsigned int img_limit
        );

        // alignment operations
        void resample_to_polar(
            const float        shift_x,
            const float        shift_y,
            const unsigned int data_idx,
            const float*       u_polar_sample_coords
        );

        void apply_FFT();

        void ccf_mult(
            const BatchHandler* ref_batch,
            const unsigned int  shift_idx,
            const unsigned int  data_idx
        );

        void apply_IFFT();

        void compute_alignment_param(
            const unsigned int            param_idx,
            const unsigned int            param_limit,
            const vector<array<float,2>>* shifts,
            AlignParam*                   aln_param
        );

        void compute_alignment_param_mref(
            const unsigned int ref_num_batch,
            const unsigned int shift_num,
            float*             u_peak_list_batch,
            float*             u_d_batch
        );

        void apply_alignment_param( AlignParam* aln_param );
        void fetch_averages( float* img_data );

        void apply_tangent_filter(
            const float cutoff_freq,
            const float falloff
        );
        
        // utility
        unsigned int size() const;
        array<unsigned int,2> ring_param() const;
        float* img_ptr( const unsigned int img_idx ) const;

        ///////////////////////////////
        float* get_tex_data(size_t* pitch) const { *pitch=img_tex_pitch; return u_img_tex_data[0]; }
        ///////////////////////////////
};

#endif
