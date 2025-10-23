# pt

This is a path tracer I created for educational purposes. It’s based on the [Ray Tracing in One Weekend](https://raytracing.github.io) book, but I’ve tried to make it faster by introducing SIMD intrinsics and multithreading. I’m using [Apple’s GCD](https://developer.apple.com/documentation/dispatch?language=objc) for multithreading. This implementation is up to 17x faster than the one in the book.

### Performance

> These numbers were recorded on an Apple M1 Pro.

Most of the heavy lifting is done by GCD, but the SIMD version alone gets a 3x improvement.

| Scene                                                                                                    | Reference | This  | Improvement |
| -------------------------------------------------------------------------------------------------------- | --------- | ----- | ----------- |
| [scene_big.inl](https://github.com/Eyoatam/pt/blob/main/scene_big.inl) at 500spp and 1200x675 resolution | 652s      | 37s   | 17x         |
| [scene_big.inl](https://github.com/Eyoatam/pt/blob/main/scene_big.inl) at 50spp and 1200x675 resolution  | 65.5s     | 3.85s | 17x         |
| [test_scene](https://github.com/Eyoatam/pt/blob/main/main.cpp#L231) at 500spp and 1200x675 resolution    | 62.15s    | 4.6s  | 13x         |
| [test_scene](https://github.com/Eyoatam/pt/blob/main/main.cpp#L231) at 50spp and 1200x675 resolution     | 6.48s     | 480ms | 13x         |
