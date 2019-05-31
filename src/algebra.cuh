/*
 * algebra.cuh
 *
 *  Created on: May 24, 2019
 *      Author: gionuno
 */

#ifndef ALGEBRA_CUH_
#define ALGEBRA_CUH_

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#define INF  1e14
#define EPS 1e-14

#define MAX_ITER 75
#define MAX_TIME 1e8
#define MIN_TIME 1e-8
#define MAX_DEPTH 3
#define __call__ __host__ __device__

struct vec
{
	double x,y,z;
	__call__ vec(double x_=0.0,double y_=0.0,double z_=0.0){ x=x_; y=y_; z=z_;}
	__call__ vec(const vec & v){ x = v.x; y = v.y; z = v.z;}
	__call__ vec & operator = (const vec & v){ if(this != &v){ x = v.x; y = v.y; z = v.z;} return *this;}
	__call__ vec & operator += (const vec & v){ x += v.x; y += v.y; z += v.z; return *this;}
	__call__ vec & operator -= (const vec & v){ x -= v.x; y -= v.y; z -= v.z; return *this;}
	__call__ vec & operator %= (const vec & v){ x *= v.x; y *= v.y; z *= v.z; return *this;}
	__call__ vec & operator *= (const double & s){ x *= s; y *= s; z *= s; return *this;}
	__call__ vec & operator /= (const vec & v){ x /= v.x; y /= v.y; z /= v.z; return *this;}
	__call__ vec & operator /= (const double & s){ x /= s; y /= s; z /= s; return *this;}
	__call__ ~vec(){}
};



inline __call__ vec operator + (const vec & a,const vec & b){return vec(a.x+b.x,a.y+b.y,a.z+b.z);}

inline __call__ vec operator - (const vec & a,const vec & b){return vec(a.x-b.x,a.y-b.y,a.z-b.z);}
inline __call__ vec operator - (const vec & a){return vec(-a.x,-a.y,-a.z);}

inline __call__ vec operator * (double s,const vec & a){return vec(s*a.x,s*a.y,s*a.z);}

inline __call__ vec operator / (const vec & a,double s){return vec(a.x/s,a.y/s,a.z/s);}
inline __call__ vec operator / (double s,const vec & a){return vec(s/a.x,s/a.y,s/a.z);}
inline __call__ vec operator / (const vec & a,const vec & b){return vec(a.x/b.x,a.y/b.y,a.z/b.z);}

inline __call__ vec operator ^ (const vec & a,const vec & b){return vec(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
inline __call__ vec operator % (const vec & a,const vec & b){return vec(a.x*b.x,a.y*b.y,a.z*b.z);}

inline __call__ vec abs(const vec & a){return vec((a.x > 0?a.x:-a.x),(a.y > 0?a.y:-a.y),(a.z > 0?a.z:-a.z));}

inline __call__ vec max(const vec & a,const vec & b){return vec((a.x > b.x?a.x:b.x),(a.y > b.y?a.y:b.y),(a.z > b.z?a.z:b.z));}
inline __call__ double max(const vec & a){return a.x > a.y?(a.x > a.z?a.x:a.z):(a.y > a.z?a.y:a.z);}

inline __call__ vec min(const vec & a,const vec & b){return vec((a.x < b.x?a.x:b.x),(a.y < b.y?a.y:b.y),(a.z < b.z?a.z:b.z));}
inline __call__ double min(const vec & a){return a.x < a.y?(a.x < a.z?a.x:a.z):(a.y < a.z?a.y:a.z);}


inline __call__ double clamp(double a,double l,double h){ return a < l?l:(a > h?h:a);}
inline __call__ vec clamp(const vec & a,const vec & l,const vec & h){return vec(clamp(a.x,l.x,h.x),clamp(a.y,l.y,h.y),clamp(a.z,l.z,h.z));}

inline __call__ double operator * (const vec & a,const vec & b){return a.x*b.x+a.y*b.y+a.z*b.z;}

inline __call__ double len(const vec & a){return sqrt(a*a);}
inline __call__ double len2(const vec & a){return a*a;}

inline __call__ vec norm(const vec & a){return a/len(a);}
inline __call__ vec reflect(const vec & a,const vec & n){return norm(a-2*(n*a)*n);}

inline __call__ vec refract(const vec & i,const vec & n,double eta_i,double eta_t)
{
    double eta = eta_i/eta_t;
    double c = i*n;
    return norm((eta*c-(c>0?1:-1)*sqrt(1+eta*(c*c-1.)))*n-eta*i);
}


#define ONES vec(1.0,1.0,1.0)
#define RED   vec(1.0,0.0,0.0)
#define GREEN vec(0.0,1.0,0.0)
#define BLUE  vec(0.0,0.0,1.0)
#define ZERO vec(0.0,0.0,0.0)

#define VMAX vec( 1e14, 1e14, 1e14)
#define VMIN vec(-1e14,-1e14,-1e14)

#define XAXIS vec(1.0,0.0,0.0)
#define YAXIS vec(0.0,1.0,0.0)
#define ZAXIS vec(0.0,0.0,1.0)

struct mat
{
	vec a,b,c;
	__call__ mat(const vec & a_=ZERO,const vec & b_=ZERO,const vec & c_=ZERO){ a = a_; b = b_; c = c_;}
	__call__ mat(const mat & m){ a = m.a; b = m.b; c = m.c;}
	__call__ ~mat(){}
	__call__ mat & operator = (const mat & m){ if(this != &m){ a = m.a; b = m.b; c = m.c;} return *this;}
};

#define EYE mat(XAXIS,YAXIS,ZAXIS)

inline __call__ mat operator + (const mat & p,const mat & q){ return mat(p.a+q.a,p.b+q.b,p.c+q.c);}
inline __call__ mat operator - (const mat & p,const mat & q){ return mat(p.a-q.a,p.b-q.b,p.c-q.c);}

inline __call__ vec operator * (const mat & m,const vec & v){ return v.x*m.a+v.y*m.b+v.z*m.c;}
inline __call__ vec operator * (const vec & v,const mat & m){ return vec(v*m.a,v*m.b,v*m.c);}

inline __call__ mat operator * (double s,const mat & m){ return mat(s*m.a,s*m.b,s*m.c);}
inline __call__ mat operator * (const mat & p,const mat & q){ return mat(p*q.a,p*q.b,p*q.c);}

inline __call__ mat inv(const mat & m)
{
	vec u = m.b^m.c;
	vec v = m.c^m.a;
	vec w = m.a^m.b;
	double d = m.a*u;
	return mat(vec(u.x/d,v.x/d,w.x/d),vec(u.y/d,v.y/d,w.y/d),vec(u.z/d,v.z/d,w.z/d));
}

inline __call__ mat trans(const mat & m)
{
	return mat(vec(m.a.x,m.b.x,m.c.x),vec(m.a.y,m.b.y,m.c.y),vec(m.a.z,m.b.z,m.c.z));
}

inline __call__ mat get_rot(double a_,const vec & q_)
{
	vec q = norm(q_);
	mat K(vec(0.0,q.z,-q.y),vec(-q.z,0.0,q.x),vec(q.y,-q.x,0.0));
	return EYE + sin(a_)*K+(1-cos(a_))*K*K;
}


/*
 * ray definidos por
 *
 * r(t) = o + t*d
 *
 * */
struct ray
{
	vec o;
	vec d;
	__call__ ray(const vec & o_ = ZERO,const vec & d_ = ZERO){ o = o_; d = d_;}
	__call__ ray(const ray & r){ o = r.o; d = r.d;}
	__call__ ray & operator = (const ray & r){ if(this != &r){ o = r.o; d = r.d;} return *this;}
	__call__ ~ray(){}
};

struct info
{
	double s;
	vec x;
	vec n;
	double eps;

	int idx;
	__call__ info(){ s = MAX_TIME; x = ZERO; n = ZERO; idx = -1; eps = EPS;}
	__call__ info(const info & I){ s = I.s; x = I.x; n = I.n; idx = I.idx; eps = I.eps;}
	__call__ ~info(){}
};

struct shape
{
	/*
	 * Objetos tienen forma canonica:
	 *     f(x)
	 *
	 * Se tendrá que aplicar transformaciones antes a la x: x -> g(x) para f(g(x))
	 * g(x) = a*(x-c);
	 */

	double a; // escala

	int type;
	vec c; // centro
	vec n; // normal
	__call__ shape(){type = -1; a = 1.0; c = ZERO; n = ZERO;}
	__call__ shape(const shape & s){ type = s.type; c = s.c; n = s.n; a = s.a;}
	__call__ ~shape(){}
};

struct samp
{
	vec w;
	vec d;
	__call__ samp(const vec & w_=ZERO,const vec & d_=ZERO){w = w_; d = d_;}
	__call__ samp(const samp & s){w = s.w; d = s.d;}
	__call__ ~samp(){}
};

inline __device__ samp phong_sample(const mat & T1,const mat & T2,
		double kd,
		double ks,
		double ex,
		curandState * state)
{
	samp ans;
	double u = curand_uniform_double(state);
	if(u < kd)
	{
		double e1 = acos(sqrt(curand_uniform_double(state)));
		double e2 = 2.0*M_PI*curand_uniform_double(state);
		double s1 = sin(e1);
		ans.d = norm(T1*vec(s1*cos(e2),s1*sin(e2),cos(e1)));
		ans.w = ((ks+kd)/kd)*ONES;
	}
	else
	{
		double e1 = pow(curand_uniform_double(state),1.0/(ex+1.0));
		double e2 = 2.0*M_PI*curand_uniform_double(state);
		double s1 = sqrt(1-e1*e1);

		ans.d = norm(T2*vec(s1*cos(e2),s1*sin(e2),e1));
		ans.w = ((ks+kd)/ks)*ONES;
	}
	return ans;
}

struct material
{
	int type;
	vec alb;
	double kd;
	double ks;
	double ex;
	__call__ material(){ alb = ONES; type = -1;kd = 0.5;ks = 0.5;ex = 0.0;}
	__call__ ~material(){}
};


#endif /* ALGEBRA_CUH_ */
