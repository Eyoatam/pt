#include <dispatch/dispatch.h>
#include <inttypes.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "vec.h"

typedef double f64;
typedef float f32;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define ArrayLength(arr) (sizeof(arr) / sizeof(arr[0]))
#define SIMD_WIDTH       (4)
#define DO_BIG_SCENE     1

#define PI (3.1415927)

static u32 xorshift32(u32 &state) {
    u32 x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;
    state = x;
    return x;
}

inline f32 degrees_to_radians(f32 deg) {
    return deg * PI / 180.0f;
}

inline f32 randomf32(u32 &seed) {
    return (xorshift32(seed) & 0xffffff) / 16777216.0f;
}

inline f32 randomf32(u32 &seed, f32 min, f32 max) {
    return min + (max - min) * randomf32(seed);
}

inline vec3 randomvec3(u32 &seed) {
    return vec3(randomf32(seed), randomf32(seed), randomf32(seed));
}

inline vec3 randomvec3(u32 &seed, f32 min, f32 max) {
    return vec3(randomf32(seed, min, max), randomf32(seed, min, max), randomf32(seed, min, max));
}

inline vec3 random_unit_vector(u32 &seed) {
    while (true) {
        vec3 p = randomvec3(seed, -1, 1);
        f32 sqn = squared_norm(p);
        if (sqn > 1e-23 && sqn <= 1.0f) return p / sqrt(sqn);
    }
}

inline vec3 random_on_hemisphere(u32 &seed, const vec3 &normal) {
    vec3 unit_on_sphere = random_unit_vector(seed);
    return (dot(unit_on_sphere, normal) > 0.0f) ? unit_on_sphere : -unit_on_sphere;
}

inline vec3 random_in_unit_disk(u32 &seed) {
    while (true) {
        vec3 p(randomf32(seed, -1, 1), randomf32(seed, -1, 1), 0.0f);
        if (squared_norm(p) < 1.0f) return p;
    }
}

inline vec3 sample_square(u32 &seed) {
    return vec3(randomf32(seed) - 0.5f, randomf32(seed) - 0.5f, 0.0f);
}

struct ray {
    vec3 origin;
    vec3 direction;
};

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
};

struct sphere {
    vec3 center;
    f32 radius;
};

struct SphereGeometry {
    SphereGeometry(u32 count) {
        size = ((count + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        centerx = new f32[size];
        centery = new f32[size];
        centerz = new f32[size];
        squared_radius = new f32[size];
        inverse_radius = new f32[size];
    }

    f32 *centerx;
    f32 *centery;
    f32 *centerz;
    f32 *squared_radius;
    f32 *inverse_radius;
    u32 size;
};

struct SphereMaterial {
    MaterialType type;
    vec3 albedo;
    f32 roughness;
    f32 refraction_index;
};

struct HitData {
    vec3 p;
    vec3 normal;
    f32 t;
};

struct Interval {
    f32 min, max;

    f32 clamp(f32 x) const {
        return (x < min) ? min : (x > max ? max : x);
    }
};

inline double linear_to_gamma(double linear_component) {
    return (linear_component > 0.0) ? sqrt(linear_component) : 0.0;
}

void write_color(FILE *out, const vec3 &pixel_color) {
    f32 r = linear_to_gamma(pixel_color.x());
    f32 g = linear_to_gamma(pixel_color.y());
    f32 b = linear_to_gamma(pixel_color.z());

    static const Interval intensity = {0.000f, 0.999f};
    fprintf(out,
            "%" PRIu8 " %" PRIu8 " %" PRIu8 "\n",
            (u8)(256 * intensity.clamp(r)),
            (u8)(256 * intensity.clamp(g)),
            (u8)(256 * intensity.clamp(b)));
}

int intersect_spheres(const SphereGeometry &spheres, const ray &r, Interval ray_t, HitData &hit) {
    vec4 hit_t = vec4(ray_t.max);
    int32x4_t id = vdupq_n_s32(-1);
    vec4 r_origx = splatx(r.origin.v);
    vec4 r_origy = splaty(r.origin.v);
    vec4 r_origz = splatz(r.origin.v);
    vec4 r_dirx = splatx(r.direction.v);
    vec4 r_diry = splaty(r.direction.v);
    vec4 r_dirz = splatz(r.direction.v);

    vec4 tmin = vec4(ray_t.min);
    int32x4_t curr_id = {0, 1, 2, 3};

    // intersect 4 spheres at once
    for (u32 i = 0; i < spheres.size; i += SIMD_WIDTH) {
        vec4 centerx = vec4(spheres.centerx + i);
        vec4 centery = vec4(spheres.centery + i);
        vec4 centerz = vec4(spheres.centerz + i);
        vec4 squared_radius = vec4(spheres.squared_radius + i);

        vec4 ocx = centerx - r_origx;
        vec4 ocy = centery - r_origy;
        vec4 ocz = centerz - r_origz;
        vec4 a = r_dirx * r_dirx + r_diry * r_diry + r_dirz * r_dirz;
        vec4 h = r_dirx * ocx + r_diry * ocy + r_dirz * ocz;
        vec4 c = ocx * ocx + ocy * ocy + ocz * ocz - squared_radius;
        vec4 discriminant = h * h - a * c;
        vec4 hit_spheres = discriminant > vec4(0.0f);

        if (any(hit_spheres)) {
            vec4 sqrtd = vec4(vsqrtq_f32(discriminant.v));
            vec4 t0 = (h - sqrtd) / a;
            vec4 t1 = (h + sqrtd) / a;
            vec4 t = select(t0 > tmin, t0, t1);

            vec4 mask = hit_spheres & (t > tmin) & (t < hit_t);
            id = select(mask, curr_id, id);
            hit_t = select(mask, t, hit_t);
        }
        curr_id = vaddq_s32(curr_id, vdupq_n_s32(SIMD_WIDTH));
    }

    // there are upto four hits; find the closest one.
    f32 min_t = hmin(hit_t);
    if (min_t < ray_t.max) { // check if the closest hit is valid
        u32 min_mask = mask(vec4(min_t) == hit_t);
        // if we have more than one closest hit, choose the one in the lowest lane.
        int lane = __builtin_ctz(min_mask); // gets the index of the first non-zero bit
        int hit_id;
        float closest_t;
        switch (lane) {
        case 0:
            hit_id = vgetq_lane_s32(id, 0);
            closest_t = vgetq_lane_f32(hit_t.v, 0);
            break;
        case 1:
            hit_id = vgetq_lane_s32(id, 1);
            closest_t = vgetq_lane_f32(hit_t.v, 1);
            break;
        case 2:
            hit_id = vgetq_lane_s32(id, 2);
            closest_t = vgetq_lane_f32(hit_t.v, 2);
            break;
        case 3:
            hit_id = vgetq_lane_s32(id, 3);
            closest_t = vgetq_lane_f32(hit_t.v, 3);
            break;
        }

        hit.p = r.origin + closest_t * r.direction;
        hit.normal =
            (hit.p -
             vec3(spheres.centerx[hit_id], spheres.centery[hit_id], spheres.centerz[hit_id])) *
            spheres.inverse_radius[hit_id];
        hit.t = closest_t;
        return hit_id;
    }
    return -1;
}

#if DO_BIG_SCENE
#include "scene_big.inl"
#else
sphere scene[] = {
    {vec3(0.0f,  -100.5f, -1.0f), 100.0f},
    {vec3(0.0f,  0.0f,    -1.2f), 0.5f  },
    {vec3(1.0f,  0.0f,    -1.0f), 0.5f  },
    {vec3(1.0f,  0.0f,    -1.0f), 0.4f  },
    {vec3(-1.0f, 0.0f,    -1.0f), 0.5f  },
    {vec3(0.0f,  0.0f,    0.0f),  0.5f  },
    {vec3(1.0f,  0.2f,    2.0f),  0.7f  },
    {vec3(0.5f,  1.0f,    0.4f),  0.3f  },
    {vec3(-3.0f, -0.2f,   2.3f),  0.8f  },
    {vec3(0.0f,  0.1f,    -2.5f), 0.8f  },
    {vec3(0.0f,  -0.1f,   3.0f),  0.2f  },
    {vec3(-1.5f, 0.2f,    0.8f),  0.25f },
};

const u32 SCENE_SIZE = ArrayLength(scene);

SphereMaterial materials[SCENE_SIZE] = {
    {METAL,      vec3(0.8f, 0.8f, 0.5f),  0.0f,  0.0f       },
    {LAMBERTIAN, vec3(0.1f, 0.2f, 0.5f),  0.0f,  0.0f       },
    {DIELECTRIC, vec3(1.0f, 1.0f, 1.0f),  0.0f,  1.5f       },
    {DIELECTRIC, vec3(1.0f, 1.0f, 1.0f),  0.0f,  1.0f / 1.5f},
    {METAL,      vec3(0.8f, 0.6f, 0.2f),  1.0f,  0.0f       },
    {METAL,      vec3(0.7f, 0.6f, 0.5f),  0.2f,  0.0f       },
    {METAL,      vec3(0.9f, 0.9f, 0.9f),  0.0f,  0.0f       },
    {LAMBERTIAN, vec3(0.9f, 0.1f, 0.1f),  0.0f,  0.0f       },
    {METAL,      vec3(0.9f, 0.9f, 0.95f), 0.05f, 0.0f       },
    {DIELECTRIC, vec3(1.0f, 1.0f, 1.0f),  0.0f,  1.33f      },
    {LAMBERTIAN, vec3(0.2f, 0.3f, 0.7f),  0.0f,  0.0f       },
    {METAL,      vec3(0.8f, 0.2f, 0.6f),  0.5f,  0.0f       },
};
#endif

SphereGeometry spheres(SCENE_SIZE);

void create_scene() {
    for (u32 i = 0; i < SCENE_SIZE; i++) {
        spheres.centerx[i] = scene[i].center.x();
        spheres.centery[i] = scene[i].center.y();
        spheres.centerz[i] = scene[i].center.z();
        spheres.inverse_radius[i] = 1.0f / scene[i].radius;
        spheres.squared_radius[i] = scene[i].radius * scene[i].radius;
    }
}

vec3 ray_color(const ray r, int depth, u32 &seed) {
    if (depth <= 0) {
        return vec3(0.0f, 0.0f, 0.0f);
    }

    Interval ray_t = {0.001, INFINITY};
    HitData hit = {};

    int hit_idx = intersect_spheres(spheres, r, ray_t, hit);
    if (hit_idx != -1) {

        switch (materials[hit_idx].type) {
        case LAMBERTIAN: {
            vec3 direction = hit.normal + random_unit_vector(seed);
            if (near_zero(direction)) {
                direction = hit.normal;
            }
            return materials[hit_idx].albedo * ray_color({hit.p, direction}, depth - 1, seed);
        }

        case METAL: {
            vec3 reflected = r.direction - 2 * dot(r.direction, hit.normal) * hit.normal;
            reflected =
                normalized(reflected) + (materials[hit_idx].roughness * random_unit_vector(seed));
            if (dot(reflected, hit.normal) > 0) {
                return materials[hit_idx].albedo * ray_color({hit.p, reflected}, depth - 1, seed);
            }
            return vec3(0.0f, 0.0f, 0.0f);
        }

        case DIELECTRIC: {
            vec3 albedo = vec3(1.0, 1.0, 1.0);
            f32 ri = 1.0 / materials[hit_idx].refraction_index;
            if (dot(hit.normal, r.direction) > 0) {
                ri = materials[hit_idx].refraction_index;
                hit.normal = -hit.normal;
            }

            vec3 unit_direction = normalized(r.direction);
            f32 cos_theta = fmin(dot(-unit_direction, hit.normal), 1.0);
            f32 sin_theta = sqrt(1 - cos_theta * cos_theta);

            // Schlick's approximation
            f32 r0 = (1 - ri) / (1 + ri);
            r0 = r0 * r0;
            f32 reflectance = r0 + (1 - r0) * pow(1 - cos_theta, 5);

            vec3 direction;
            if ((ri * sin_theta > 1.0) || (reflectance > randomf32(seed))) {
                // total internal reflection
                direction = unit_direction - 2 * dot(unit_direction, hit.normal) * hit.normal;
            } else { // refract
                vec3 r_out_perp = ri * (unit_direction + cos_theta * hit.normal);
                vec3 r_out_parallel = -sqrt(fabs(1.0 - squared_norm(r_out_perp))) * hit.normal;
                direction = r_out_perp + r_out_parallel;
            }

            return albedo * ray_color({hit.p, direction}, depth - 1, seed);
        }
        }
    }
    f32 a = 0.5f * (normalized(r.direction).y() + 1.0f);
    return (1.0f - a) * vec3{1.0f, 1.0f, 1.0f} + a * vec3{0.5f, 0.7f, 1.0f};
}

int main() {
    f32 aspect_ratio = 16.0f / 9.0f;
    u32 image_width = 1200;
    u32 samples_per_pixel = 500;
    int max_depth = 50;

    f32 vfov = 20.0;
    vec3 lookfrom = vec3(13.0, 2.0, 3.0);
    vec3 lookat = vec3(0.0, 0.0, 0.0);
    vec3 vup = vec3(0.0, 1.0, 0.0);
    vec3 camera_center = lookfrom;

    f32 defocus_angle = 0.6;
    f32 focus_dist = 10.0;

    u32 image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    f32 theta = degrees_to_radians(vfov);
    f32 viewport_height = 2.0 * tan(theta / 2) * focus_dist;
    f32 viewport_width = viewport_height * ((f32)image_width / image_height);

    vec3 w = normalized(lookfrom - lookat);
    vec3 u = normalized(cross(vup, w));
    vec3 v = cross(w, u);

    vec3 viewport_u = viewport_width * u;
    vec3 viewport_v = viewport_height * -v;

    vec3 pixel_delta_u = viewport_u / image_width;
    vec3 pixel_delta_v = viewport_v / image_height;

    vec3 viewport_upper_left = camera_center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
    vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    f32 defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
    vec3 defocus_disk_u = u * defocus_radius;
    vec3 defocus_disk_v = v * defocus_radius;

    create_scene();

    vec3 *image_buffer = new vec3[image_width * image_height];

    // create a serial queue safely updating progress
    __block u32 lines_completed = 0;
    dispatch_queue_t progress_queue = dispatch_queue_create("progress", DISPATCH_QUEUE_SERIAL);

    // use concurrent queue for parallel execution
    dispatch_queue_t render_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    dispatch_group_t group = dispatch_group_create();

    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    u64 begin_time = mach_absolute_time();
    // dispatch work for each row
    for (u32 j = 0; j < image_height; j++) {
        dispatch_group_async(group, render_queue, ^{
          for (u32 i = 0; i < image_width; i++) {
              u32 seed = 2463534242 + j * image_width + i;
              vec3 pixel_color = vec3(0.0, 0.0, 0.0);

              for (u32 sample = 0; sample < samples_per_pixel; sample++) {
                  vec3 offset = sample_square(seed);
                  vec3 pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) +
                                      ((j + offset.y()) * pixel_delta_v);
                  vec3 p = random_in_unit_disk(seed);
                  vec3 defocused_center =
                      camera_center + (p.x() * defocus_disk_u) + (p.y() * defocus_disk_v);
                  vec3 origin = (defocus_angle <= 0) ? camera_center : defocused_center;
                  vec3 ray_direction = pixel_sample - origin;
                  pixel_color += ray_color((ray){origin, ray_direction}, max_depth, seed);
              }
              image_buffer[j * image_width + i] = pixel_color / samples_per_pixel;
          }

          // update progress
          dispatch_async(progress_queue, ^{
            lines_completed++;
            if (lines_completed % 10 == 0) {
                fprintf(stderr,
                        "\rProgress: %u/%u scanlines (%.1f%%)    ",
                        lines_completed,
                        image_height,
                        100.0f * lines_completed / image_height);
            }
          });
        });
    }

    // wait for all work to complete
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);

    u64 end_time = mach_absolute_time();
    f64 render_time = ((f64)(end_time - begin_time) * ((f64)info.numer / info.denom)) * 1e-6;

    begin_time = mach_absolute_time();
    // write output
    fprintf(stdout, "P3\n%u %u\n255\n", image_width, image_height);
    for (u32 j = 0; j < image_height; j++) {
        for (u32 i = 0; i < image_width; i++) {
            write_color(stdout, image_buffer[j * image_width + i]);
        }
    }
    end_time = mach_absolute_time();
    f64 output_time = ((f64)(end_time - begin_time) * ((f64)info.numer / info.denom)) / 1e6;
    f64 total_time = render_time + output_time;
    fprintf(stderr, "\n--------------------------------\n");
    fprintf(stderr, "Total time: %.2lfms (%.2lfs)\n", total_time, total_time * 0.001);
    fprintf(stderr, "   Rendering: %.2lfms (%.2lfs)\n", render_time, render_time * 0.001);
    fprintf(stderr, "   Writing output: %.2lfms (%.2lfs)\n", output_time, output_time * 0.001);
}
