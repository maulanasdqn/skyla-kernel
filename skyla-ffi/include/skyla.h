#ifndef SKYLA_H
#define SKYLA_H

#include <stdbool.h>
#include <stdint.h>

typedef struct Skyla Skyla;

typedef struct {
    uint32_t prompt_tokens;
    double prefill_ms;
    uint32_t tokens;
    double tokens_per_second;
    uint32_t position;
    uint32_t context;
} SkylaStats;

typedef void (*SkylaTokenCallback)(void *user, const char *text, bool thinking);

Skyla *skyla_open(const char *path, uint32_t context, char **error);
const char *skyla_model_name(const Skyla *s);
char *skyla_device_name(const Skyla *s);
bool skyla_chat(Skyla *s, const char *message, const char *system, float temperature, float top_p,
                uint32_t max_tokens, SkylaTokenCallback callback, void *user, SkylaStats *stats,
                char **error);
void skyla_cancel(const Skyla *s);
void skyla_reset(Skyla *s);
void skyla_free(Skyla *s);
void skyla_string_free(char *p);

#endif
