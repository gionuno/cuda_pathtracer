/*
 * shape.cuh
 *
 *  Created on: May 26, 2019
 *      Author: gionuno
 */

#ifndef SHAPE_CUH_
#define SHAPE_CUH_

#include "algebra.cuh"
#include "material.cuh"

__call__ double eval_plane(const vec & p,const vec & n){ return n*p;}
__call__ double eval_sphere(const vec & p){return len(p)-1.0;}
__call__ double eval_box(const vec & p,const vec & n){ vec d = abs(p)-n; return len(max(d,ZERO))+min(max(d),0.0);}
__call__ double eval_torus(const vec & p,const vec & n)
{
	vec q = vec(len(vec(p.x,p.y))-n.x,p.z);
	return len(q)-n.y;
}
__call__ double eval_cylin(const vec & p,const vec & n){ return len(vec(p.x-n.x,p.y-n.y,0.0))-n.z;}

__call__ double inter_plane(const vec & n,const vec & c, const ray & r)
{
	double num = n*(c-r.o);
	double den = n*r.d;
	if(abs(den) < EPS)
		return MAX_TIME;
	else if(num/den < MIN_TIME)
		return MAX_TIME;
	return num/den;
}

__call__ double inter_sphere(const vec & c,double b,const ray & r)
{
	vec oc = r.o-c;
	double len2oc = len2(oc);
	double dtoc   = r.d*oc;

	double det = b*b-len2oc+dtoc*dtoc;
	if(det < 0.0) return MAX_TIME;

	det = sqrt(det);
	double t1 = -dtoc-det;
	double t2 = -dtoc+det;
	if(t1 > MIN_TIME) return t1 < MAX_TIME ? t1 : MAX_TIME;
	else if(t2 > MIN_TIME) return t2 < MAX_TIME ? t2 : MAX_TIME;
	else return MAX_TIME;
}

__call__ double eval(const shape & s,const vec & p)
{
	double ans = INF;
	vec q = (p-s.c)/s.a;
	switch(s.type)
	{
		case 2: //box
			ans = eval_box(q,s.n);
		break;
		case 3: //torus
			ans = eval_torus(q,s.n);
		break;
	}
	return ans >= INF? INF: s.a*ans;
}

__call__ vec grad(const shape & s,const vec & p,const vec & r)
{
	vec ans;

	vec qa_p = (p+EPS*XAXIS-s.c)/s.a;
	vec qa_m = (p-EPS*XAXIS-s.c)/s.a;
	vec qb_p = (p+EPS*YAXIS-s.c)/s.a;
	vec qb_m = (p-EPS*YAXIS-s.c)/s.a;
	vec qc_p = (p+EPS*ZAXIS-s.c)/s.a;
	vec qc_m = (p-EPS*ZAXIS-s.c)/s.a;
	switch(s.type)
	{
		case 2: //box
			ans = vec(eval_box(qa_p,s.n)-eval_box(qa_m,s.n),
					  eval_box(qb_p,s.n)-eval_box(qb_m,s.n),
					  eval_box(qc_p,s.n)-eval_box(qc_m,s.n));
		break;

		case 3: //torus
			ans = vec(eval_torus(qa_p,s.n)-eval_torus(qa_m,s.n),
					  eval_torus(qb_p,s.n)-eval_torus(qb_m,s.n),
					  eval_torus(qc_p,s.n)-eval_torus(qc_m,s.n));
		break;
	}
	if(s.type < 2) return ans;
	double lans = len(ans);
	if(lans == 0.0) return -r;
	return ans/lans;
}

__call__ void dist_search(int nshapes,shape * shapes, const vec & p,double & rad, int & rdx)
{
	rdx =  -1;
	rad = INF;
	for(int n=0;n<nshapes;n++)
	if(shapes[n].type >= 2)
	{
		double aux_rad = eval(shapes[n],p);
		if(rad > aux_rad)
		{
			rad = aux_rad;
			rdx = n;
		}
	}
}

__call__ double direct_search(int nshapes,shape * shapes,const ray & r,info & I)
{
	double t   = MAX_TIME;
	int    tdx = -1;
	for(int n=0;n<nshapes;n++)
	if(shapes[n].type < 2)
	{
		double aux_t = MAX_TIME;
		switch(shapes[n].type)
		{
			case 0: //plane
				aux_t = inter_plane(shapes[n].n,shapes[n].c,r);
			break;
			case 1: //sphere
				aux_t = inter_sphere(shapes[n].c,shapes[n].a,r);
			break;
		}
		if(t > aux_t)
		{
			t = aux_t;
			tdx = n;
		}
	}
	if(t < MAX_TIME && tdx >= 0)
	{
		I.idx = tdx;
		I.x = r.o+t*r.d;
		I.s = t;
		I.eps = 1e-4;
		switch(shapes[tdx].type)
		{
			case 0:
				I.n = shapes[tdx].n;
			break;
			case 1:
				I.n = norm(I.x-shapes[tdx].c);
			break;
		}
	}
	return t;
}

__call__ double intersect(int nshapes,shape * shapes,const ray & r, info & I)
{

	double direct_t = direct_search(nshapes,shapes,r,I);
	double t        = MIN_TIME;

	int rdx    =  -1;
	double rad = INF;
	int i=0;
	for(i=0;i<MAX_ITER && t < direct_t;i++)
	{
		rad = INF;
		rdx =  -1;
		dist_search(nshapes,shapes,r.o+t*r.d,rad,rdx);

		if(rad < 1e-8)
			break;
		t += rad;
	}
	if(i == MAX_ITER || t > direct_t)
		return direct_t;
	else if(t <= direct_t)
	{
		I.idx = rdx;
		I.eps = 2.0*max(abs(rad),1e-4);
		I.s   = t;
		I.x   = r.o+I.s*r.d;
		I.n   = grad(shapes[rdx],I.x,r.d);
		return t;
	}
	/*
	float omega = 1.2;

	float cand_E = INF;
	float cand_t = 2*EPS;

	float t      = 2*EPS;

	double prev_rad = 0.0;
	double step  = 0.0;


	double fsign = rad > 0.0 ? 1.0:-1.0;

	int i = 0;

	for(i=0;i<100;i++)
	{
		rdx =  -1;
		rad = INF;
		for(int n=0;n<nshapes;n++)
		{
				double aux_rad = eval(shapes[n],r.o+t*r.d);
				if(rad > aux_rad)
				{
					rad = aux_rad;
					rdx = n;
				}
		}

		double srad = fsign*rad;
		rad = abs(srad);

		bool sorFail = (omega > 1.0) && (rad+prev_rad < step);
		if(sorFail)
		{
			step  -= omega * step;
			omega = 1.0;
		}
		else
			step   = omega * srad;

		prev_rad = rad;
		float E = rad / t;
		if(!sorFail && E < cand_E)
		{
			cand_t = t;
			cand_E = E;
		}
		if( (!sorFail && E < 1e-10) || t >= INF) break;

		t += step;
	}
	if(t < INF)
	{
		I.idx = rdx;
		I.eps = 2.0*max(abs(rad),1e-14);
		I.s   = cand_t;
		I.x   = r.o+I.s*r.d;
		I.n   = grad(shapes[rdx],I.x,r.d);
	}*/
	return direct_t;
}



#endif /* SHAPE_CUH_ */
