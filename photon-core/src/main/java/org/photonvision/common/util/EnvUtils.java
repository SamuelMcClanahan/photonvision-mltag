/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.common.util;

import java.util.Locale;

/** Helpers for optional runtime tuning flags used by platform-specific integrations. */
public final class EnvUtils {
    private EnvUtils() {}

    public static boolean hasValue(String name) {
        String raw = System.getenv(name);
        return raw != null && !raw.isBlank();
    }

    public static String string(String name, String defaultValue) {
        String raw = System.getenv(name);
        return raw == null || raw.isBlank() ? defaultValue : raw.trim();
    }

    public static boolean flag(String name) {
        return flag(name, false);
    }

    public static boolean flag(String name, boolean defaultValue) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }

        return switch (raw.trim().toLowerCase(Locale.ROOT)) {
            case "1", "true", "yes", "on" -> true;
            case "0", "false", "no", "off" -> false;
            default -> defaultValue;
        };
    }

    public static int integer(String name, int defaultValue) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }

        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    public static int nonNegativeInteger(String name, int defaultValue) {
        return Math.max(0, integer(name, defaultValue));
    }

    public static double decimal(String name, double defaultValue) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }

        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    public static double positiveDecimal(String name, double defaultValue) {
        double value = decimal(name, defaultValue);
        return value > 0 ? value : defaultValue;
    }
}
