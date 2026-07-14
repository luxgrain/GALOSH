/* avs_dump.c — run an AviSynth script and dump all frames as raw planar
 * YUV (Y then U then V per frame) to a file.  Used by the SMDegrain
 * thSAD calibration and AVS-baseline benches (ffmpeg here lacks the
 * avisynth demuxer; the C API route needs no extra deps).
 *
 * Usage: avs_dump.exe script.avs out.raw [max_frames]
 * Prints "W H CW CH N" on stdout for the reader.
 *
 * Build: gcc -O2 avs_dump.c C:/Windows/System32/AviSynth.dll
 *        -I<avisynth_c.h dir> -o avs_dump.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include "avisynth_c.h"

int main(int argc, char **argv)
{
  if(argc < 3) { fprintf(stderr, "usage: %s script.avs out.raw [max]\n", argv[0]); return 2; }
  AVS_ScriptEnvironment *env = avs_create_script_environment(6);
  AVS_Value res = avs_invoke(env, "Import",
                             avs_new_value_string(argv[1]), NULL);
  if(avs_is_error(res)) { fprintf(stderr, "AVS error: %s\n", avs_as_string(res)); return 1; }
  AVS_Clip *clip = avs_take_clip(res, env);
  avs_release_value(res);
  const AVS_VideoInfo *vi = avs_get_video_info(clip);
  if(!avs_is_planar(vi)) { fprintf(stderr, "not planar\n"); return 1; }

  int n = vi->num_frames;
  if(argc > 3) { int m = atoi(argv[3]); if(m > 0 && m < n) n = m; }
  const int w = vi->width, h = vi->height;
  /* chroma dims from the first frame's pitches/row sizes */
  FILE *fo = fopen(argv[2], "wb");
  if(!fo) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }

  int cw = 0, chh = 0;
  for(int i = 0; i < n; i++)
  {
    AVS_VideoFrame *f = avs_get_frame(clip, i);
    if(!f) { fprintf(stderr, "frame %d NULL\n", i); return 1; }
    static const int planes[3] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };
    for(int p = 0; p < 3; p++)
    {
      const unsigned char *ptr = avs_get_read_ptr_p(f, planes[p]);
      const int pitch = avs_get_pitch_p(f, planes[p]);
      const int rs = avs_get_row_size_p(f, planes[p]);
      const int ph = avs_get_height_p(f, planes[p]);
      if(p == 1 && i == 0) { cw = rs; chh = ph; }
      for(int y = 0; y < ph; y++)
        fwrite(ptr + (size_t)y * pitch, 1, rs, fo);
    }
    avs_release_video_frame(f);
  }
  fclose(fo);
  printf("%d %d %d %d %d\n", w, h, cw, chh, n);
  avs_release_clip(clip);
  avs_delete_script_environment(env);
  return 0;
}
