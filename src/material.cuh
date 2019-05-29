/*
 * material.cuh
 *
 *  Created on: May 26, 2019
 *      Author: gionuno
 */

#ifndef MATERIAL_CUH_
#define MATERIAL_CUH_

#include "algebra.cuh"

__device__ samp get_samp(const material & m,const ray & r,const info & I,curandState * state)
{
	samp ans;
	ans.w = m.alb;
	ans.e = (m.type == 0)? m.alb : ZERO;
	//if(m.type == 0) return ans; //Light
	if(m.type == 0 || m.type == 1) //Lambertian Diffuse
	{
		mat T;
		T.a = norm((abs(I.n.x) > 0.1?YAXIS:XAXIS)^I.n);
		T.b = norm(I.n^T.a);
		T.c = I.n;
		ans.d = hemi_uniform(T,state);
	}
	if(m.type == 2) //Specular
	{
		ans.d = reflect(r.d,I.n);
	}
	return ans;
}


#endif /* MATERIAL_CUH_ */
