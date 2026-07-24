#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "rotary_encoder.h"

#define TAG "rotary_encoder"

#define ROT_ENC_A_GPIO 27
#define ROT_ENC_B_GPIO 26

#define ENABLE_HALF_STEPS false  // Set to true to enable tracking of rotary encoder at half step resolution

// Количество шагов энкодера на один полный оборот (уточнить для конкретной модели)
#define STEPS_PER_REVOLUTION   3600.0f

// Длина окружности колеса/вала, вращаемого энкодером, в мм
#define WHEEL_CIRCUMFERENCE_MM  8.57232L

// Интервал между замерами скорости, мс
#define MEASURE_INTERVAL_MS 200

void app_main(void)
{
    ESP_ERROR_CHECK(gpio_install_isr_service(0));

    rotary_encoder_state_t state = { 0 };
    rotary_encoder_info_t info = { 0 };

    ESP_ERROR_CHECK(rotary_encoder_init(&info, ROT_ENC_A_GPIO, ROT_ENC_B_GPIO));
    ESP_ERROR_CHECK(rotary_encoder_enable_half_steps(&info, ENABLE_HALF_STEPS));
    ESP_ERROR_CHECK(rotary_encoder_reset(&info));

    int32_t prev_position = 0;
    int64_t prev_time_us = 0;

    while (1)
    {
        vTaskDelay(MEASURE_INTERVAL_MS / portTICK_PERIOD_MS);

        ESP_ERROR_CHECK(rotary_encoder_get_state(&info, &state));

        int64_t now_us = esp_timer_get_time();
        int32_t delta_position = state.position - prev_position;
        int64_t delta_time_us = now_us - prev_time_us;

        float speed_rps = 0.0f;
        float speed_mms = 0.0f;

        if (delta_time_us > 0 && prev_time_us != 0) {
            float delta_time_s = (float)delta_time_us / 1000000.0f;
            speed_rps = ((float)delta_position / STEPS_PER_REVOLUTION) / delta_time_s;
            speed_mms = speed_rps * WHEEL_CIRCUMFERENCE_MM;
        }

        prev_position = state.position;
        prev_time_us = now_us;

        // Единственная телеметрическая строка на измерение - чистый JSON.
        // Именно её на стороне ПК читает SpeedReader в main.py и достаёт
        // из неё только поле "speed_mms" (скорость конвейера, мм/с).
        printf("{\"pos\":%ld,\"dir\":%d,\"t_us\":%lld,\"speed_rps\":%.3f,\"speed_mms\":%.2f}\n",
               (long)state.position*(-1),
               (int)state.direction,
               (long long)now_us,
               speed_rps*(-1),
               speed_mms*(-1));
    }

    ESP_ERROR_CHECK(rotary_encoder_uninit(&info));
}