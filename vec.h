#include <arm_neon.h>
#include <math.h>

typedef double f64;
typedef float f32;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// this is heavily inspired by: http:/www.codersnotes.com/notes/maths-lib-2016/

struct vec3 {
    vec3() {
    }
    explicit vec3(f32 x, f32 y, f32 z) {
        f32 r[4] = {x, y, z, 0.0f};
        v = vld1q_f32(r);
    }

    inline f32 x() const {
        return vgetq_lane_f32(v, 0);
    }
    inline f32 y() const {
        return vgetq_lane_f32(v, 1);
    }
    inline f32 z() const {
        return vgetq_lane_f32(v, 2);
    }

    float32x4_t v;
};

inline vec3 operator-(vec3 a) {
    a.v = vnegq_f32(a.v);
    return a;
}

inline vec3 operator+(vec3 a, const vec3 b) {
    a.v = vaddq_f32(a.v, b.v);
    return a;
}
inline vec3 operator-(vec3 a, const vec3 b) {
    a.v = vsubq_f32(a.v, b.v);
    return a;
}
inline vec3 operator*(vec3 a, const vec3 b) {
    a.v = vmulq_f32(a.v, b.v);
    return a;
}
inline vec3 operator*(vec3 a, const f32 b) {
    a.v = vmulq_n_f32(a.v, b);
    return a;
}
inline vec3 operator*(const f32 b, vec3 a) {
    a.v = vmulq_n_f32(a.v, b);
    return a;
}
inline vec3 operator/(const vec3 &a, const f32 b) {
    return a * (1 / b);
}
inline void operator+=(vec3 &a, const vec3 b) {
    a = a + b;
}
inline void operator-=(vec3 &a, const vec3 b) {
    a = a - b;
}
inline void operator*=(vec3 &a, const f32 b) {
    a = a * b;
}
inline void operator/=(vec3 &a, const f32 b) {
    a = a / b;
}

inline f32 sum(const vec3 a) {
    return a.x() + a.y() + a.z();
}
inline f32 dot(const vec3 a, const vec3 b) {
    return sum(a * b);
}
inline vec3 cross(const vec3 a, const vec3 b) {
    f32 ax = a.x();
    f32 ay = a.y();
    f32 az = a.z();
    f32 bx = b.x();
    f32 by = b.y();
    f32 bz = b.z();

    return vec3(ay * bz - by * az, //
                az * bx - ax * bz, //
                ax * by - ay * bx);
}

inline f32 norm(const vec3 a) {
    return sqrtf(dot(a, a));
}
inline f32 squared_norm(const vec3 a) {
    return dot(a, a);
}

inline vec3 min(vec3 a, const vec3 b) {
    a.v = vminq_f32(a.v, b.v);
    return a;
}
inline vec3 max(vec3 a, const vec3 b) {
    a.v = vmaxq_f32(a.v, b.v);
    return a;
}

inline bool near_zero(const vec3 &v) {
    float32x4_t abs_v = vabsq_f32(v.v);
    return vmaxvq_f32(abs_v) < 1e-8f;
}

struct vec4 {
    vec4() {
    }
    explicit inline vec4(const f32 *p) {
        v = vld1q_f32(p);
    }
    explicit vec4(f32 x, f32 y, f32 z, f32 w) {
        f32 r[4] = {x, y, z, w};
        v = vld1q_f32(r);
    }
    explicit vec4(f32 a) {
        v = vdupq_n_f32(a);
    }
    explicit vec4(float32x4_t a) {
        v = a;
    }

    inline f32 x() const {
        return vgetq_lane_f32(v, 0);
    }
    inline f32 y() const {
        return vgetq_lane_f32(v, 1);
    }
    inline f32 z() const {
        return vgetq_lane_f32(v, 2);
    }
    inline f32 w() const {
        return vgetq_lane_f32(v, 3);
    }
    float32x4_t v;
};

inline vec4 operator-(vec4 a) {
    a.v = vnegq_f32(a.v);
    return a;
}

inline vec4 operator+(vec4 a, const vec4 b) {
    a.v = vaddq_f32(a.v, b.v);
    return a;
}
inline vec4 operator-(vec4 a, const vec4 b) {
    a.v = vsubq_f32(a.v, b.v);
    return a;
}
inline vec4 operator*(vec4 a, const vec4 b) {
    a.v = vmulq_f32(a.v, b.v);
    return a;
}
inline vec4 operator*(vec4 a, const f32 b) {
    a.v = vmulq_n_f32(a.v, b);
    return a;
}
inline vec4 operator*(const f32 b, vec4 a) {
    a.v = vmulq_n_f32(a.v, b);
    return a;
}
inline vec4 operator/(vec4 a, const vec4 b) {
    a.v = vdivq_f32(a.v, b.v);
    return a;
}
inline vec4 operator/(const vec4 a, const f32 b) {
    return a * (1 / b);
}
inline void operator+=(vec4 &a, const vec4 b) {
    a = a + b;
}
inline void operator-=(vec4 &a, const vec4 b) {
    a = a - b;
}
inline void operator*=(vec4 &a, const f32 b) {
    a = a * b;
}
inline void operator/=(vec4 &a, const f32 b) {
    a = a / b;
}

inline vec4 operator==(vec4 a, vec4 b) {
    a.v = vceqq_f32(a.v, b.v);
    return a;
}
inline vec4 operator!=(vec4 a, vec4 b) {
    a.v = vmvnq_u32(vceqq_f32(a.v, b.v));
    return a;
}
inline vec4 operator<(vec4 a, vec4 b) {
    a.v = vcltq_f32(a.v, b.v);
    return a;
}
inline vec4 operator>(vec4 a, vec4 b) {
    a.v = vcgtq_f32(a.v, b.v);
    return a;
}
inline vec4 operator<=(vec4 a, vec4 b) {
    a.v = vcleq_f32(a.v, b.v);
    return a;
}
inline vec4 operator>=(vec4 a, vec4 b) {
    a.v = vcgeq_f32(a.v, b.v);
    return a;
}
inline vec4 operator&(vec4 a, vec4 b) {
    a.v = vandq_u32(a.v, b.v);
    return a;
}
inline vec4 operator|(vec4 a, vec4 b) {
    a.v = vorrq_u32(a.v, b.v);
    return a;
}
inline vec4 min(vec4 a, vec4 b) {
    a.v = vminq_f32(a.v, b.v);
    return a;
}
inline vec4 max(vec4 a, vec4 b) {
    a.v = vmaxq_f32(a.v, b.v);
    return a;
}
inline vec3 normalized(const vec3 a) {
    return a / norm(a);
}

inline f32 hmin(vec4 a) {
    return vminvq_f32(a.v);
}

// returns a 4-bit code where bit0..bit3 is X..W
// `a` is expected to be the result of a comparison operation.
inline u32 mask(vec4 a) {
    constexpr uint32x4_t movemask = {1, 1 << 1, 1 << 2, 1 << 3};
    uint32x4_t t0 = vreinterpretq_u32_f32(a.v);
    uint32x4_t t1 = vandq_u32(t0, movemask);
    return vaddvq_u32(t1);
}

inline u32 any(vec4 v) {
    return mask(v) != 0;
}

// (mask & a) | (~mask & b)
inline vec4 select(vec4 mask, vec4 a, vec4 b) {
    a.v = vbslq_f32(mask.v, a.v, b.v);
    return a;
}
inline int32x4_t select(vec4 mask, int32x4_t a, int32x4_t b) {
    return vbslq_s32(mask.v, a, b);
}

inline vec4 splatx(float32x4_t v) {
    return vec4(vdupq_lane_f32(vget_low_f32(v), 0));
}
inline vec4 splaty(float32x4_t v) {
    return vec4(vdupq_lane_f32(vget_low_f32(v), 1));
}
inline vec4 splatz(float32x4_t v) {
    return vec4(vdupq_lane_f32(vget_high_f32(v), 0));
}
inline vec4 splatw(float32x4_t v) {
    return vec4(vdupq_lane_f32(vget_high_f32(v), 1));
}
