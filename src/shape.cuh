/*
 * shape.cuh
 *
 *  Created on: May 26, 2019
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
__call__ double eval_cylin(const vec & p)
{
	return len(vec(p.x,p.z,0.0))-1.0;

}

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
		case 0: //plane
			ans = eval_plane(q,s.n);
		break;
		case 1: //sphere
			ans = eval_sphere(q);
		break;
		case 2: //box
			ans = eval_box(q,s.n);
		break;
		case 3: //torus
			ans = eval_torus(q,s.n);
		break;
		case 4: //cylin
			ans = eval_cylin(q);
		break;
	}
	return ans >= INF? INF: s.a*ans;
}

__call__ vec grad(const shape & s,const vec & p,const vec & r)
{
	vec ans = -r;

	vec qa = (p+EPS*vec( 1.0,-1.0,-1.0)-s.c)/s.a;
	vec qb = (p+EPS*vec(-1.0,-1.0, 1.0)-s.c)/s.a;
	vec qc = (p+EPS*vec(-1.0, 1.0,-1.0)-s.c)/s.a;
	vec qd = (p+EPS*vec( 1.0, 1.0, 1.0)-s.c)/s.a;

	double ea,eb,ec,ed;
	switch(s.type)
	{
		case 0: //plane
			ea = eval_plane(qa,s.n);
			eb = eval_plane(qb,s.n);
			ec = eval_plane(qc,s.n);
			ed = eval_plane(qd,s.n);
		break;
		case 1: //sphere
			ea = eval_sphere(qa);
			eb = eval_sphere(qb);
			ec = eval_sphere(qc);
			ed = eval_sphere(qd);
		break;
		case 2: //box
			ea = eval_box(qa,s.n);
			eb = eval_box(qb,s.n);
			ec = eval_box(qc,s.n);
			ed = eval_box(qd,s.n);
		break;

		case 3: //torus
			ea = eval_torus(qa,s.n);
			eb = eval_torus(qb,s.n);
			ec = eval_torus(qc,s.n);
			ed = eval_torus(qd,s.n);
		break;

		case 4: //cylinder
			ea = eval_cylin(qa);
			eb = eval_cylin(qb);
			ec = eval_cylin(qc);
			ed = eval_cylin(qd);
		break;
	}
	if(s.type < 2) return ans;
	ans =  ea*vec( 1.0,-1.0,-1.0)
		  +eb*vec(-1.0,-1.0, 1.0)
		  +ec*vec(-1.0, 1.0,-1.0)
		  +ed*vec( 1.0, 1.0, 1.0);
	double lans = len(ans);
	if(lans == 0.0) return -r;
	return ans/lans;
}

__call__ void dist_search(int nshapes,shape * shapes,const vec & p,double & rad, int & rdx)
{
	rdx =  -1;
	rad = INF;
	for(int n=0;n<nshapes;n++)
	//if(shapes[n].type >= 2)
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
		I.eps = 2*EPS;
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
	int i=0;

	double direct_t = direct_search(nshapes,shapes,r,I);
	double        t = MIN_TIME;

	int   rdx      =  -1;
	double rad      = INF;
	dist_search(nshapes,shapes,r.o,rad,rdx);

	/*
	double omega = 1.2;

	double prev_rad = 0.0;
	double step     = 0.0;
	double cand_E = INF;
	double cand_t = direct_t;
	int    cand_rdx =  -1;
	double cand_rad = INF;

	double fsign = rad > 0.0 ? 1.0:-1.0;
	*/
	for(i=0;i<MAX_ITER && t < direct_t;i++)
	{
		rad = INF;
		rdx =  -1;
		dist_search(nshapes,shapes,r.o+t*r.d,rad,rdx);

		if(abs(rad) < 1e-8)
			break;
		t += rad;
	}

	if(i == MAX_ITER || t > direct_t)
		return MAX_ITER;
	else if(t < direct_t)
	{
		I.idx = rdx;
		I.eps = 2.0*max(abs(rad),1e-4);
		I.s   = t;
		I.x   = r.o+I.s*r.d;
		I.n   = grad(shapes[rdx],I.x,r.d);
		return i;
	}

/*	double tlast = 0.0;
	double radlast = rad;
	int    rdxlast = rdx;

	for(i=0;i<MAX_ITER && t < direct_t;i++)
	{
		rdx =  -1;
		rad = INF;
		dist_search(nshapes,shapes,r.o+t*r.d,rad,rdx);
		if(abs(rad) < 1e-8) break;

		step = -rad*(t-tlast)/(rad-radlast);
		t = tlast + step;

		tlast   = t;
		radlast = rad;
		rdxlast = rdx;
	}
	if(i == MAX_ITER || t > direct_t || t < MIN_TIME)
		return MAX_ITER;
	else if(MIN_TIME < t && t < direct_t)
	{
		I.idx = rdx;
		I.eps = 2.0*max(abs(rad),1e-4);
		I.s   = t;
		I.x   = r.o+I.s*r.d;
		I.n   = grad(shapes[rdx],I.x,r.d);
		return i;
	}*/
	return MAX_ITER;
}



#endif /* SHAPE_CUH_ */
