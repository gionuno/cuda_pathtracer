/*
 * material.cuh
 *
 *  Created on: May 26, 2019
 */

#ifndef MATERIAL_CUH_
#define MATERIAL_CUH_

#include "algebra.cuh"

__device__ samp get_samp(const material & m,const ray & r,const info & I,curandState * state)
{
	samp ans;
	ans.w = m.alb;
	//if(m.type == 0) return ans; //Light
	if(m.type == 1) //Phong + Diffuse
	{
		mat T1;
		T1.a = norm((abs(I.n.x) > 0.1?YAXIS:XAXIS)^I.n);
		T1.b = norm(I.n^T1.a);
		T1.c = I.n;

		vec r2 = reflect(r.d,I.n);
		mat T2;
		T2.a = norm((abs(r2.x) > 0.1?YAXIS:XAXIS)^r2);
		T2.b = norm(r2^T2.a);
		T2.c = r2;

		samp ans2 = phong_sample(T1,T2,m.kd,m.ks,m.ex,state);
		ans.w %= ans2.w;
		ans.d  = ans2.d;
	}
	if(m.type == 2) //Specular
	{
		ans.d = reflect(r.d,I.n);
	}
	return ans;
}


#endif /* MATERIAL_CUH_ */
