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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipeline;

public record StreamDemand(boolean input, boolean output) {
    public static final StreamDemand NONE = new StreamDemand(false, false);
    public static final StreamDemand ALWAYS = new StreamDemand(true, true);

    private boolean isExplicitDemand() {
        return this != NONE && this != ALWAYS;
    }

    public boolean effectiveInput(CVPipelineSettings settings) {
        return input && (settings.inputShouldShow || isExplicitDemand());
    }

    public boolean effectiveOutput(CVPipelineSettings settings) {
        return output && (settings.outputShouldShow || isExplicitDemand());
    }

    public boolean effectiveOutputDraw(AdvancedPipelineSettings settings) {
        return effectiveOutput(settings) && (settings.outputShouldDraw || isExplicitDemand());
    }

    public boolean anyEffective(AdvancedPipelineSettings settings) {
        return effectiveInput(settings) || effectiveOutput(settings) || effectiveOutputDraw(settings);
    }
}
