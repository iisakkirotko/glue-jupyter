<template>
    <div>
        <div class="text-subtitle-2 font-weight-bold">Color</div>
        <div>
            <v-select label="color" :items="cmap_mode_items" v-model="cmap_mode_selected" hide-details />
        </div>
        <template v-if="glue_state.cmap_mode === 'Linear'">
            <div>
                <v-select label="attribute" :items="cmap_att_items" v-model="cmap_att_selected" hide-details />
            </div>
            <div>
                <glue-float-field label="min" :value.sync="glue_state.cmap_vmin" />
            </div>
            <div>
                <glue-float-field label="max" :value.sync="glue_state.cmap_vmax" />
            </div>
            <div>
                <v-select label="colormap" :items="cmap_items" :value="glue_state.cmap" @change="set_colormap" hide-details/>
            </div>
        </template>
        <!-- <div>
            <v-subheader class="pl-0 slider-label">opacity</v-subheader>
            <glue-throttled-slider wait="300" min="0" max="1" step="0.01" :value.sync="glue_state.alpha" hide-details />
        </div> -->
        <div class="text-subtitle-2 font-weight-bold">Points</div>
        <template v-if="glue_state.markers_visible">
            <div>
                <v-select label="size" :items="size_mode_items" v-model="size_mode_selected" hide-details />
            </div>
            <template v-if="glue_state.size_mode === 'Linear'">
                <div>
                    <v-select label="attribute" :items="size_att_items" v-model="size_att_selected" hide-details />
                </div>
                <div>
                    <glue-float-field label="min" :value.sync="glue_state.size_vmin" />
                </div>
                <div>
                    <glue-float-field label="max" :value.sync="glue_state.size_vmax" />
                </div>
            </template>
            <template>
                <div>
                    <v-subheader class="pl-0 slider-label">size</v-subheader>
                    <glue-throttled-slider wait="300" min="0.1" max="10" step="0.01" :value.sync="glue_state.size"
                        hide-details />
                </div>
                <div>
                    <v-subheader class="pl-0 slider-label">size scaling</v-subheader>
                    <glue-throttled-slider wait="300" min="0.1" max="10" step="0.01" :value.sync="glue_state.size_scaling"
                        hide-details />
                </div>
            </template>
        </template>
    </div>
</template>
<script>
</script>
