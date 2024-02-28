<template>
  <div class="px-5">
    <a-drawer v-model:visible="protein_visible" title="Protein Viewer"
      :closable="false" :forceRender="true" placement="left" width="750">
      <MolStar v-if="viewer == 'molstar'" :pdb="current_view_pdb" />
      <ProteinViewer v-else :pdb="current_view_pdb" />
    </a-drawer>
    <a-drawer v-model:visible="request_visible" title="Request Editor"
      :closable="false" placement="right" size="large">
      <Request :request="current_request_json" :on_load="on_load_request" />
    </a-drawer>
    <a-modal v-model:visible="comment_edit_visible"
      :title="'Comments for ' + current_comment_record.name"
      @ok="update_comment" okType="ghost" :forceRender="true">
      <input id="modal-comment" v-model="current_comment"
        placeholder="Describe this record with several words."
        class="outline-none p-2 w-full border border-gray-300" />
    </a-modal>
    <a-modal v-model:visible="tags_edit_visible" title="Tags" @ok="update_tags"
      okType="ghost" :forceRender="true">
      <div class="flex items-end justify-between">
        <div class="flex items-center">
          <div class="mx-2">Edit mode:</div>
          <a-select class="w-24" v-model:value="current_tags_edit_mode">
            <a-select-option v-for="v in tags_edit_modes" :value="v" :key="v">{{
            v
            }}</a-select-option>
          </a-select>
        </div>
        <p class="text-gray-400 text-xs">separated by ,</p>
      </div>
      <input id="modal-tags" v-model="current_tags"
        class="mt-3 outline-none p-2 w-full border border-gray-300" />
    </a-modal>
    <div class="flex justify-center mt-5">
      <div>
        <div v-if="filter_sender == 'cameo'">
          <div class="flex">
            <a-col class="mr-8">
              <a-statistic-countdown title="Receive" :value="receive_time"
                format="D [days]  HH [h]  mm [m]  ss [s]">
              </a-statistic-countdown>
            </a-col>
            <a-col>
              <a-statistic-countdown title="Submit" :value="casp_ddl"
                format="D [days]  HH [h]  mm [m]  ss [s]" />
            </a-col>
          </div>
          <div class="flex justify-center mt-5">
            <div class="w-32 border-b flex items-center text-gray-500">
              <input v-model="cameo_to_date" class="
                  placeholder:italic placeholder:text-slate-400
                  bg-white
                  w-full
                  py-2
                  pl-2
                  pr-3
                  focus:outline-none
                  sm:text-sm
                " placeholder="CAMEO to date" type="text"
                name="came_to_date" />
            </div>
            <a-button class="ml-5" @click="download_cameo"
              :disabled="downloading_cameo" type="primary" ghost>
              Download CAMEO PDBs
            </a-button>
          </div>
        </div>
        <div v-else-if="filter_sender == 'casp'">
          <div class="flex text-left">
            <div class="font-bold">
              <div class="px-3 border-b py-3 bg-gray-100">Date</div>
              <div class="px-3 mt-2 py-1">Targets</div>
            </div>
            <div v-for="targets in recent_casp_targets"
              :key="'t' + targets.date" class="text-right">
              <div class="text-gray-700 px-3 border-b py-3 bg-gray-100">
                {{ targets.date }}
              </div>
              <div class="px-3 py-1 mt-2 flex flex-col pr-3"
                v-bind:class="{ 'font-bold': targets.remain == 0 }">
                <span class="cursor-pointer underline"
                  @click="filter_name = t.Target" v-for="t in targets.targets"
                  :key="'t_name_' + t.Target">{{ t.Target }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <a-divider></a-divider>

    <div class="mt-5 flex justify-between">
      <div>
        <div class="flex mb-2 items-center">
          <a-switch v-model:checked="show_comment" size="default" />
          <div class="mx-2">Show Comment</div>
          <div class="mx-2">Viewer:</div>
          <a-select class="w-20" v-model:value="viewer">
            <a-select-option v-for="v in viewers" :value="v" :key="v">{{
            v
            }}</a-select-option>
          </a-select>
        </div>
        <div class="flex items-end">
          <a-button :disabled="!has_selected" :loading="loading_rerun"
            @click="rerun_requests" type="primary" ghost>
            Rerun
          </a-button>
          <a-button class="ml-2" :disabled="!has_selected"
            :loading="loading_email" @click="email_requests" type="primary"
            ghost>
            Email
          </a-button>
          <a-button class="ml-2" :disabled="!has_selected" @click="update_plddt"
            :loading="loading_plddt" type="primary" ghost>
            pLDDT
          </a-button>
          <a-button class="ml-2" :disabled="!has_selected" @click="update_lddt"
            :loading="loading_lddt" type="primary" ghost>
            LDDT
          </a-button>
          <a-button class="ml-2" :disabled="!has_selected"
            @click="analyze_requests" type="primary" ghost>
            Analyze
          </a-button>
          <a-button class="ml-2" :disabled="!has_selected"
            @click="edit_tags(null, selected_hash_ids)" :loading="updating_tags"
            type="primary" ghost>
            Tag
          </a-button>
          <span class="ml-5 py-1">
            <template v-if="has_selected">
              {{ `Selected ${selected_hash_ids.length} items` }}
            </template>
          </span>
        </div>
      </div>

      <div class="flex flex-col items-end">
        <div class="flex items-center">
          <div class="mb-2">Hide Removed</div>
          <a-switch class="mb-2 mx-2" v-model:checked="hide_invisible"
            size="default" />
          <a-radio-group v-model:value="filter_sender" class="mb-2 ml-2">
            <a-radio-button value="all">All</a-radio-button>
            <a-radio-button value="cameo">CAMEO</a-radio-button>
            <a-radio-button value="casp15">CASP15</a-radio-button>
            <a-radio-button value="other">Other</a-radio-button>
          </a-radio-group>
        </div>
        <a-radio-group v-model:value="filter_date">
          <a-radio-button value="all">All</a-radio-button>
          <a-radio-button value="week">Last Week</a-radio-button>
          <a-radio-button value="month">Last Month</a-radio-button>
          <a-radio-button value="year">Last Year</a-radio-button>
        </a-radio-group>
      </div>
    </div>

    <div class="flex justify-between items-center mt-5">
      <div class="w-96 border-b flex items-center">
        <input v-model="filter_name" class="
            placeholder:italic placeholder:text-slate-400
            bg-white
            w-full
            py-2
            pl-2
            pr-3
            focus:outline-none
            sm:text-sm
          " placeholder="Filter Name ..." type="text" name="search" />
        <close-outlined @click="filter_name = ''" class="cursor-pointer" />
        <a-tag @click="
          filter_mode =
            filter_modes[
              (filter_modes.indexOf(filter_mode) + 1) % filter_modes.length
            ]
        " class="ml-2 cursor-pointer w-16 text-center">
          {{ filter_mode }}
        </a-tag>
        <a-popover placement="bottom">
          <template #content>
            <div class="text-xs text-gray-500">
              <p>
                1. Name search: filter records by name matching, e.g. T1190.
              </p>
              <p>2. Tag search: filter records by tags, e.g. tag:0822,0813.</p>
              <p>
                3. Sender search: filter records by senders, e.g.
                sender:cameo,casp15.
              </p>
              <p>
                4. all/primary: whether to show only records whose name does not
                contain ___.
              </p>
            </div>
          </template>
          <question-circle-outlined />
        </a-popover>
      </div>
      <div class="flex items-center mt-3">
        <span class="mr-2 text-xs text-gray-400">
          {{
          "Last refresh: " +
          moment(last_refresh).format("YYYY-MM-DD HH:mm:ss")
          }}
        </span>
        <a-tooltip placement="topRight">
          <template #title>
            <span>Auto Refresh per {{ refresh_interval }} seconds.</span>
          </template>
          <a-switch class="m-1" v-model:checked="is_auto_refresh"
            size="default" />
        </a-tooltip>
        <div @click="get_requests"
          class="cursor-pointer m-1 flex items-center text-lg">
          <loading-outlined v-if="loading_requests" />
          <sync-outlined v-else />
        </div>

        <a-divider class="m-1" type="vertical" />

        <div class="">
          <div class="flex justify-end text-lg">
            <a-tooltip placement="top">
              <template #title>
                <span>Protein Viewer</span>
              </template>
              <monitor-outlined @click="protein_visible = true"
                class="m-1 cursor-pointer" />
            </a-tooltip>
            <a-tooltip placement="top">
              <template #title>
                <span>Request Editor</span>
              </template>
              <form-outlined @click="request_visible = true"
                class="m-1 cursor-pointer" />
            </a-tooltip>
          </div>
        </div>
      </div>
    </div>

    <div class="mt-5">
      <a-table :dataSource="filtered_data" :columns="columns" :row-selection="{
        selectedRowKeys: selected_hash_ids,
        onChange: on_select_change,
      }" :pagination="{ defaultPageSize: 25 }" size="middle" :row-class-name="
        (record, _index) =>
          (days_to_ddl(record) == 0
            ? 'bg-purple-50'
            : days_to_ddl(record) == 1
            ? 'bg-yellow-50'
            : '') + (record.visible == 0 ? ' opacity-50' : '')
      ">
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'last_update_time'">
            <div class="flex items-center">
              <a-popover title="History States" placement="right">
                <template #content>
                  <a-timeline :reverse="true"
                    class="p-2 pb-0 overflow-auto max-h-96">
                    <a-timeline-item
                      v-if="record.state_msg.length > max_show_history">
                      ...
                    </a-timeline-item>
                    <a-timeline-item v-for="(state, index) in record.state_msg.slice(
                      1 - max_show_history
                    )" :key="record.hash_id + state" class="!pb-2">
                      <template #dot
                        v-if="index == record.state_msg.length - 1">
                        <clock-circle-outlined style="font-size: 16px" />
                      </template>
                      <a-popover placement="right">
                        <template #content>
                          <p class="
                              text-xs
                              flex-wrap
                              text-wrap
                              max-w-lg
                              overflow-x-auto
                            ">
                            {{
                            moment(state.time).format("YYYY-MM-DD HH:mm:ss")
                            }}
                          </p>
                        </template>
                        <span class="cursor-default">
                          {{ moment(state.time).fromNow() }}
                        </span>
                      </a-popover>

                      <a-tag class="ml-2 mt-1" :color="
                        state.state.includes('ERROR')
                          ? 'volcano'
                          : state.state.includes('START')
                          ? 'blue'
                          : 'green'
                      ">
                        {{ state.state }}
                      </a-tag>
                    </a-timeline-item>
                  </a-timeline>
                </template>
                <a-tag :color="
                  record.state.includes('ERROR')
                    ? 'volcano'
                    : record.state.includes('START')
                    ? 'blue'
                    : 'green'
                " class="cursor-pointer">
                  {{ record.state }}
                </a-tag>
              </a-popover>
              <a-popover placement="right" v-if="record.error">
                <template #content>
                  <p
                    class="text-xs flex-wrap text-wrap max-w-lg overflow-x-auto">
                    {{ record.error }}
                  </p>
                </template>
                <warning-outlined class="text-red-700" :class="{
                  'opacity-30': !record.state.includes('ERROR'),
                  'opacity-100': record.state.includes('ERROR'),
                }" />
              </a-popover>
            </div>
            <div class="text-xs text-gray-400 ml-1 mt-1">
              <a-popover placement="right">
                <template #content>
                  <p
                    class="text-xs flex-wrap text-wrap max-w-lg overflow-x-auto">
                    Latest update:
                    {{
                    moment(record.last_update_time).format(
                    "YYYY-MM-DD HH:mm:ss"
                    )
                    }}
                  </p>
                </template>
                <span class="cursor-default">
                  <span class="text-gray-500">{{
                  from_now_time_abbr(record.last_update_time)
                  }}</span>
                  <span class="text-gray-400"> / {{ total_time(record) }}</span>
                </span>
              </a-popover>
            </div>
          </template>
          <template v-else-if="column.key === 'name'">
            <div>
              <div class="flex items-center">
                <a-popover title="Config" placement="right">
                  <template #content>
                    <div class="flex max-w-lg max-h-96 overflow-auto text-xs">
                      <pre>{{
                        JSON.stringify(record.request_json, null, 2)
                      }}</pre>
                    </div>
                  </template>
                  <span class="
                      inline-flex
                      justify-between
                      items-center
                      cursor-pointer
                    ">
                    <span class="cursor-pointer hover:underline"
                      v-if="record.sender == 'cameo'"
                      @click="open_cameo_url(record.name)">{{ record.name
                      }}</span>
                    <span v-else> {{ record.name }} </span>
                  </span>
                </a-popover>
                <copy-outlined class="
                    cursor-pointer
                    ml-1
                    text-xs text-gray-300
                    hover:text-gray-500
                  " @click="copy_text(record.name)" />
                <a-tooltip placement="top">
                  <template #title>
                    <span>Edit the comment.</span>
                  </template>
                  <form-outlined class="
                      cursor-pointer
                      ml-1
                      text-xs text-gray-300
                      hover:text-gray-500
                    " @click="edit_comment(record)" />
                </a-tooltip>
              </div>
              <div class="w-64">
                <div v-if="record.tags && record.tags.length > 0">
                  <a-tag class="mr-1 cursor-pointer text-xs"
                    v-for="tag in record.tags" :key="record.hash_id + tag"
                    @click="filter_name = 'tag:' + tag" color="blue">
                    <span>{{ tag }}</span>
                  </a-tag>
                </div>
                <a-tooltip placement="topLeft"
                  v-if="show_comment && record.reserved.comment">
                  <template #title>
                    <span>Edit the comment.</span>
                  </template>
                  <span class="
                      text-gray-400
                      mb-1
                      text-xs
                      flex
                      items-baseline
                      cursor-pointer
                    " @click="edit_comment(record)">
                    <span>
                      {{ record.reserved.comment }}
                    </span>
                  </span>
                </a-tooltip>
              </div>
            </div>
          </template>
          <template v-else-if="column.key === 'L'">
            <div class="flex">
              <p class="w-12">
                {{ record.request_json.sequence.length }}
              </p>
            </div>
          </template>
          <template v-else-if="column.key === 'analysis'">
            <div class="flex flex-wrap items-center">
              <a-tooltip>
                <template #title>
                  <span>Click to copy the sequence.</span>
                </template>
                <a-tag class="cursor-pointer m-1" color="purple"
                  @click="copy_sequence(record)">
                  Seq
                </a-tag>
              </a-tooltip>
              <a-popover
                v-if="record.reserved && record.reserved.selected_template_info"
                placement="right" title="Template">
                <template #content>
                  <div class="text-xs overflow-auto max-h-96">
                    <div>
                      <p class="font-bold mb-2">Selected Templates</p>
                      <div v-if="record.reserved.selected_template_info">
                        <div v-for="(temp, i) in record.reserved
                        .selected_template_info" :key="record.name + temp[0]"
                          class="flex">
                          <p class="w-7 text-right">{{ i + 1 }}. &nbsp;</p>
                          <p class="w-16 font-mono">
                            <a :href="
                              'https://www.rcsb.org/3d-sequence/' +
                              temp[0].split('_')[0].toUpperCase()
                            " target="_blank">{{ temp[0].toUpperCase() }}</a>
                          </p>
                          <p class="text-gray-400 text-right w-12">
                            {{ parseFloat(temp[1]).toFixed(2) }}
                          </p>
                        </div>
                      </div>
                    </div>
                    <div class="mt-5">
                      <p class="font-bold mb-2">20 Filtered Templates</p>
                      <div v-if="record.reserved.template_info">
                        <div v-for="(temp, i) in record.reserved.template_info"
                          :key="record.name + temp[0]" class="flex">
                          <p class="w-7 text-right">{{ i + 1 }}. &nbsp;</p>
                          <p class="w-16 font-mono">
                            <a :href="
                              'https://www.rcsb.org/3d-sequence/' +
                              temp[0].split('_')[0].toUpperCase()
                            " target="_blank">{{ temp[0].toUpperCase() }}</a>
                          </p>
                          <p class="text-gray-400 text-right w-12">
                            {{ parseFloat(temp[1]).toFixed(2) }}
                          </p>
                        </div>
                      </div>
                    </div>
                    <div class="mt-5">
                      <p class="font-bold mb-2">All Searched Templates</p>
                      <div v-if="record.reserved.searched_template_info">
                        <div v-for="(temp, i) in record.reserved
                        .searched_template_info" :key="record.name + temp[0]"
                          class="flex">
                          <p class="w-7 text-right">{{ i + 1 }}. &nbsp;</p>
                          <p class="w-16 font-mono">
                            <a :href="
                              'https://www.rcsb.org/3d-sequence/' +
                              temp[0].split('_')[0].toUpperCase()
                            " target="_blank">{{ temp[0].toUpperCase() }}</a>
                          </p>
                          <p class="text-gray-400 text-right w-12">
                            {{ temp[1] }}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </template>
                <a-tag class="cursor-pointer m-1" color="purple">
                  Template
                </a-tag>
              </a-popover>
              <a-popover
                v-if="record.path_tree && record.path_tree.alphafold.input_a3m"
                placement="bottom" title="MSA Coverage">
                <template #content>
                  <GuidesFigure :src="get_msa_coverage(record)" :img_width="600"
                    :max_num="record.request_json.sequence.length"
                    :ruler_visible="false" :init_ruler_offset="60" />
                </template>
                <a-tag class="cursor-pointer m-1" color="purple"
                  @click="open_text(record.path_tree.alphafold.input_a3m)">
                  MSA
                </a-tag>
              </a-popover>
              <a-popover v-if="record.plddt" title="pLDDT" placement="bottom">
                <template #content>
                  <GuidesFigure :src="get_plddt_all(record)" :img_width="600"
                    :max_num="record.request_json.sequence.length"
                    :ruler_visible="false" :init_ruler_offset="60" />
                </template>
                <a-tag class="cursor-pointer m-1"
                  :color="record.plddt ? 'purple' : 'default'">
                  pLDDT
                </a-tag>
              </a-popover>
              <a-popover v-if="record.lddt" title="lDDT" placement="bottom">
                <template #content>
                  <GuidesFigure :src="get_lddt_all(record)" :img_width="600"
                    :max_num="record.request_json.sequence.length"
                    :ruler_visible="false" :init_ruler_offset="60" />
                </template>
                <a-tag class="cursor-pointer m-1"
                  :color="record.lddt ? 'purple' : 'default'">
                  lDDT
                </a-tag>
              </a-popover>
              <a-tag v-if="record.reserved && record.reserved.exp_pdb_path"
                class="cursor-pointer m-1"
                :color="record.reserved.exp_pdb_path ? 'purple' : 'default'"
                @click="show_pdb(record, record.reserved.exp_pdb_path)">
                PDB
              </a-tag>
              <a-tooltip>
                <template #title>
                  <span>Focus on this protein.</span>
                </template>
                <a-tag class="cursor-pointer m-1 flex" color="purple"
                  @click="filter_name = record.name">
                  <aim-outlined class="py-1" />
                </a-tag>
              </a-tooltip>
              <a-tooltip>
                <template #title>
                  <span>Open analysis page of this record.</span>
                </template>
                <a-tag class="cursor-pointer m-1 flex" color="purple"
                  @click="analyze_request(record)">
                  <search-outlined class="py-1" />
                </a-tag>
              </a-tooltip>
              <a-tooltip>
                <template #title>
                  <span>Open analysis page of this protein.</span>
                </template>
                <a-tag class="cursor-pointer m-1 flex items-center"
                  color="purple" @click="analyze_request(record, 'name')">
                  <search-outlined class="py-1" />
                  *
                </a-tag>
              </a-tooltip>
            </div>
          </template>
          <template v-else-if="column.key === 'plddt'">
            <div class="flex flex-wrap"
              v-if="record.path_tree && record.path_tree.alphafold">
              <div v-for="pdb_info in parse_pdbs_info(record)"
                :key="record.name + pdb_info.model_name + 'lddt'">
                <a-tag class="m-1 cursor-pointer" :color="
                  pdb_info.plddt < 80
                    ? 'volcano'
                    : pdb_info.plddt < 90
                    ? 'orange'
                    : 'green'
                " @click="show_pdb(record, pdb_info.relaxed_pdb)"
                  @dblclick="show_pdb(record, pdb_info.relaxed_pdb, true)">
                  <span>{{ pdb_info.model_name }}:</span>
                  <span>&nbsp;{{ pdb_info.plddt.toFixed(2) }}</span>
                </a-tag>
                <hr v-if="record.lddt" class="my-1" />
                <div class="flex justify-center" v-if="record.lddt">
                  <a-tag class="m-1 cursor-pointer" :color="
                    pdb_info.lddt < 80
                      ? 'volcano'
                      : pdb_info.lddt < 90
                      ? 'orange'
                      : 'green'
                  ">
                    <span>{{ pdb_info.model_name }}:</span>
                    <span>&nbsp;{{ pdb_info.lddt.toFixed(2) }}</span>
                  </a-tag>
                </div>
              </div>
            </div>
          </template>
          <template v-else-if="column.key === 'action'">
            <div class="flex justify-center">
              <a-tooltip>
                <template #title>
                  <span>Stop running.</span>
                </template>
                <pause-circle-outlined @click="stop_running(record)"
                  class="mx-1 cursor-pointer" />
              </a-tooltip>
              <a-tooltip>
                <template #title>
                  <span>Modify and submit a new request.</span>
                </template>
                <edit-outlined @click="edit_config(record)"
                  class="mx-1 cursor-pointer" />
              </a-tooltip>
              <a-tooltip v-if="record.visible == 1">
                <template #title>
                  <span>Remove this record.</span>
                </template>
                <delete-outlined @click="set_visibility(record, false)"
                  class="mx-1 cursor-pointer text-red-700" />
              </a-tooltip>
              <a-tooltip v-else>
                <template #title>
                  <span>Restore this record.</span>
                </template>
                <undo-outlined @click="set_visibility(record, true)"
                  class="mx-1 cursor-pointer" />
              </a-tooltip>
            </div>
          </template>
        </template>
      </a-table>
    </div>
  </div>
</template>

<script>
import { API } from "@/js/api";
import { Data } from "@/js/data";
import axios from "axios";
import { filter, keyBy, forEach, split, assign, join } from "lodash";
import {
  ClockCircleOutlined,
  CopyOutlined,
  EditOutlined,
  SyncOutlined,
  LoadingOutlined,
  MonitorOutlined,
  SearchOutlined,
  FormOutlined,
  CloseOutlined,
  DeleteOutlined,
  UndoOutlined,
  AimOutlined,
  WarningOutlined,
  PauseCircleOutlined,
  QuestionCircleOutlined,
} from "@ant-design/icons-vue";
import moment from "moment";
import { message } from "ant-design-vue";
import Request from "../components/Request.vue";
import GuidesFigure from "../components/GuidesFigure.vue";

export default {
  name: "HomeView",
  components: {
    ClockCircleOutlined,
    CopyOutlined,
    EditOutlined,
    SyncOutlined,
    FormOutlined,
    MonitorOutlined,
    LoadingOutlined,
    SearchOutlined,
    CloseOutlined,
    DeleteOutlined,
    UndoOutlined,
    AimOutlined,
    WarningOutlined,
    PauseCircleOutlined,
    QuestionCircleOutlined,
    Request,
    GuidesFigure,
  },

  mounted: function () {
    this.get_requests();
    this.get_cameo_data();
    this.get_casp_data();
    this.today = moment();
    setInterval(() => {
      if (
        this.is_auto_refresh &&
        !document.hidden &&
        moment().diff(this.last_refresh, "seconds") > this.refresh_interval
      ) {
        this.get_requests();
        this.today = moment();
      }
    }, 10000);
    this.bind_event();
  },

  data() {
    return {
      // data
      requests: [],
      requests_map: {},
      selected_hash_ids: [],
      filter_date: "month",
      filter_sender: "cameo",
      filter_name: "",
      filter_mode: "all",
      filter_modes: ["all", "primary"],
      hide_invisible: true,
      cameo_data: {},
      casp_data: {},
      today: {},

      // state
      loading_rerun: false,
      loading_requests: false,
      loading_email: false,
      loading_lddt: false,
      loading_plddt: false,
      show_comment: true,
      comment_edit_visible: false,
      current_comment_record: {},
      current_comment: "",
      tags_edit_visible: false,
      current_tags: "",
      current_tags_hash_ids: null,
      tags_edit_modes: ["add", "remove", "replace"],
      current_tags_edit_mode: "add",
      updating_tags: false,
      is_auto_refresh: true,
      last_refresh: {},
      refresh_interval: 120, // seconds
      max_show_history: 48,
      loading_cameo: false,
      downloading_cameo: false,
      cameo_to_date: moment().format("YYYY-MM-DD"),
      request_visible: false,
      current_request_json: {},
      protein_visible: false,
      current_view_pdb: null,

      viewer: "pv",
      viewers: ["pv", "molstar"],

      columns: [
        // {
        //   title: "State",
        //   dataIndex: "state",
        //   key: "state",
        // },
        {
          title: "Latest State",
          dataIndex: "last_update_time",
          key: "last_update_time",
          sorter: (a, b) =>
            new Date(a.last_update_time) - new Date(b.last_update_time),
          defaultSortOrder: "descend",
        },
        {
          title: "Name",
          dataIndex: "name",
          key: "name",
          sorter: (a, b) => a.name.localeCompare(b.name),
        },
        {
          title: "Length",
          key: "L",
          sorter: (a, b) =>
            a.request_json.sequence.length > b.request_json.sequence.length,
        },
        {
          title: "Analysis",
          key: "analysis",
        },
        {
          title: "pLDDT / lDDT",
          dataIndex: "plddt",
          key: "plddt",
        },
        {
          title: "Actions",
          key: "action",
        },
      ],
    };
  },

  computed: {
    has_selected() {
      return this.selected_hash_ids.length > 0;
    },

    parsed_data() {
      return Data.parse_data(this.requests);
    },

    filtered_data() {
      let days = 0;
      let data = this.parsed_data;
      if (this.filter_date == "three") {
        days = 3;
      } else if (this.filter_date == "week") {
        days = 7;
      } else if (this.filter_date == "month") {
        days = 30;
      } else if (this.filter_date == "year") {
        days = 365;
      }

      if (days > 0) {
        data = filter(data, (record) => {
          let diff_days = moment().diff(moment(record.release_date), "days");
          return diff_days < days;
        });
      }

      if (this.filter_sender == "cameo" || this.filter_sender == "casp15") {
        data = filter(data, (record) => {
          return record.sender == this.filter_sender;
        });
      } else if (this.filter_sender == "other") {
        data = filter(data, (record) => {
          return (
            record.sender != "cameo" && record.sender != "casp15"
          );
        });
      }

      if (this.filter_mode == "primary") {
        data = filter(data, (record) => {
          return !record.name.toString().includes("___");
        });
      }

      if (this.filter_name != "") {
        data = filter(data, (record) => {
          if (this.filter_name.startsWith("tag:")) {
            let tags = this.filter_name.substring(4).split(",");
            for (let tag of tags) {
              if (record.reserved.tags && record.reserved.tags.includes(tag)) {
                return true;
              }
            }
            return false;
          } else if (this.filter_name.startsWith("sender:")) {
            let senders = this.filter_name.substring(7).split(",");
            for (let sender of senders) {
              if (record.sender.includes(sender)) {
                return true;
              }
            }
          } else {
            return record.name
              .toString()
              .includes(this.pure_name(this.filter_name));
          }
        });
      }

      if (this.hide_invisible) {
        data = filter(data, (record) => {
          return record.visible == 1;
        });
      }

      return data;
    },

    receive_time() {
      var d = new Date();
      d.setDate(d.getDate() + ((6 + 7 - d.getDay()) % 7 || 7));
      d.setUTCHours(0, 0, 0);
      return d.toString();
    },

    casp_ddl() {
      var d = new Date();
      d.setDate(d.getDate() + ((3 + 7 - d.getDay()) % 7 || 7));
      d.setUTCHours(0, 0, 0);
      return d.toString();
    },

    recent_casp_targets() {
      let targets = [];
      if (this.today.add) {
        for (let i = -2; i < 7; i++) {
          let date = moment(this.today)
            .add(i, "days")
            .subtract(6, "hours")
            .format("YYYY-MM-DD");
          targets.push({
            remain: i,
            date: date,
            targets: this.casp_data.by_date ? this.casp_data.by_date[date] : [],
          });
        }
      }
      return targets;
    },
  },

  methods: {
    moment,
    bind_event: function () {
      // document
      //   .getElementById("modal_comment")
      //   .addEventListener("keyup", (e) => {
      //     // if is enter, then click ok
      //     console.log(e)
      //   });
    },
    // APIs
    get_requests: function () {
      this.loading_requests = true;
      axios(API.get_requests())
        .then((response) => {
          this.requests = Data.add_key(response.data);
          this.requests_map = keyBy(this.requests, "hash_id");
          this.last_refresh = new Date();
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.loading_requests = false;
        });
    },

    rerun_requests: function () {
      this.loading_rerun = true;
      axios(API.rerun_hashids(this.selected_hash_ids))
        .then((response) => {
          this.update_states(response.data);
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.loading_rerun = false;
          this.selected_hash_ids = [];
        });
    },

    email_requests: function () {
      this.loading_email = true;
      axios(API.email_hashids(this.selected_hash_ids))
        .then((response) => {
          this.update_states(response.data);
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.loading_email = false;
          this.selected_hash_ids = [];
        });
    },

    analyze_requests: function () {
      let res = this.$router.resolve({
        path: "/analyze",
        query: { hash_ids: join(this.selected_hash_ids, ",") },
      });
      window.open(res.href, "_blank");
      this.selected_hash_ids = [];
    },

    analyze_request: function (record, mode) {
      let query = {};
      if (mode == "name") {
        query.name = record.name.split("___")[0];
      } else {
        query.hash_ids = record.hash_id;
      }
      let res = this.$router.resolve({
        path: "/analyze",
        query: query,
      });
      window.open(res.href, "_blank");
    },

    update_lddt: function () {
      this.loading_lddt = true;
      axios(API.update_lddt_hashids(this.selected_hash_ids))
        .then((response) => {
          this.update_states(response.data);
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.loading_lddt = false;
          this.selected_hash_ids = [];
        });
    },

    update_plddt: function () {
      this.loading_plddt = true;
      axios(API.update_plddt_hashids(this.selected_hash_ids))
        .then((response) => {
          this.update_states(response.data);
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.loading_plddt = false;
          this.selected_hash_ids = [];
        });
    },

    edit_tags: function (record, hash_ids) {
      this.current_tags = "";
      if (record) {
        if (record.reserved && record.reserved.tags) {
          this.current_tags = record.reserved.tags;
        }
        this.current_tags_hash_ids = record.hash_id;
        this.current_tags_edit_mode = "replace";
      } else {
        this.current_tags_hash_ids = hash_ids;
        this.current_tags_edit_mode = "add";
      }
      this.tags_edit_visible = true;
    },

    update_tag: function (hash_id, tags, mode) {
      this.updating_tags = true;
      axios(API.update_tags(hash_id, tags, mode))
        .then((response) => {
          this.update_states(response.data);
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.updating_tags = false;
        });
    },

    update_tags: function () {
      if (this.current_tags_hash_ids) {
        this.updating_tags = true;
        axios(
          API.update_tags(
            this.current_tags_hash_ids,
            this.current_tags,
            this.current_tags_edit_mode
          )
        )
          .then((response) => {
            this.update_states(response.data);
          })
          .catch((error) => {
            console.log(error);
          })
          .finally(() => {
            this.updating_tags = false;
            if (Array.isArray(this.current_tags_hash_ids)) {
              this.selected_hash_ids = [];
            }
            this.tags_edit_visible = false;
          });
      }
    },

    get_cameo_data: function () {
      this.loading_cameo = true;
      axios(API.get_cameo_data(moment().format("YYYY-MM-DD")))
        .then((response) => {
          forEach(response.data.aaData, (record) => {
            this.cameo_data[record.target] = record;
          });
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.loading_cameo = false;
        });
    },

    download_cameo: function () {
      this.download_cameo = true;
      axios(API.download_cameo(moment().format("YYYY-MM-DD")))
        .then((response) => {
          message.success(
            "CAMEO PDB downloader status: " + response.data.status
          );
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.download_cameo = false;
        });
    },

    get_casp_data: function () {
      axios(API.get_casp_data())
        .then((response) => {
          this.casp_data.by_target = keyBy(response.data, "Target");
          this.casp_data.by_date = Data.parse_casp_data(response.data);
        })
        .catch((error) => {
          console.log(error);
        });
    },

    parse_pdbs_info: function (record) {
      return Data.parse_pdbs_info(record);
    },

    get_msa_coverage: function (record) {
      if (record.path_tree && record.path_tree.alphafold) {
        return API.image_url(record.path_tree.alphafold.msa_coverage_image);
      }
      return null;
    },

    get_plddt_all: function (record) {
      if (record.path_tree && record.path_tree.alphafold) {
        return API.image_url(record.path_tree.alphafold.plddt_image);
      }
      return null;
    },

    get_lddt_all: function (record) {
      if (record.path_tree && record.path_tree.alphafold) {
        var plddt_path = record.path_tree.alphafold.plddt_image;
        var lddt_path = plddt_path.replace("predicted_LDDT", "LDDT");
        return API.image_url(lddt_path);
      }
      return null;
    },

    open_cameo_url: function (name) {
      var cameo_obj = {};
      // slice 2022-04-02_00000009_2 from 2022-04-02_00000009_2_127
      name = name.split("___")[0];
      name = name.slice(0, -4);
      if (name in this.cameo_data) {
        cameo_obj = this.cameo_data[name];
      }
      if (cameo_obj.pdbid) {
        window.open(API.cameo_url(cameo_obj), "_blank");
      } else {
        if (this.loading_cameo) {
          message.loading("CAMEO data is loading... Wait for seconds.");
        } else {
          message.warning("Not found in recent release of CAMEO.");
        }
      }
    },

    open_text: function (text_path) {
      window.open(API.pdb_url(text_path), "_blank");
    },

    copy_text: function (text) {
      this.$copyText(text).then(
        function () {
          message.success("Copied to clipboard.");
        },
        function () {
          message.warning("Copy failed.");
        }
      );
    },

    copy_sequence: function (record) {
      let sequence = record.request_json.sequence;
      this.copy_text(sequence);
    },

    copy_config: function (record) {
      var request_json_text = JSON.stringify(
        Data.parse_config(record),
        null,
        2
      );
      this.copy_text(request_json_text);
    },

    edit_comment: function (record) {
      this.current_comment_record = record;
      this.current_comment = "";
      if (record.reserved && record.reserved.comment) {
        this.current_comment = record.reserved.comment;
      }
      this.comment_edit_visible = true;
    },

    update_comment: function () {
      let comment_changed = false;
      if (this.current_comment) {
        if (
          !this.current_comment_record.reserved ||
          !this.current_comment_record.reserved.comment
        ) {
          comment_changed = true;
        } else if (
          this.current_comment != this.current_comment_record.reserved.comment
        ) {
          comment_changed = true;
        }
      } else if (
        this.current_comment_record.reserved &&
        this.current_comment_record.reserved.comment
      ) {
        comment_changed = true;
      }
      if (comment_changed) {
        axios(
          API.update_comment(
            this.current_comment_record.hash_id,
            this.current_comment
          )
        )
          .then((response) => {
            this.update_states(response.data);
          })
          .catch((error) => {
            console.log(error);
          });
      }
      this.comment_edit_visible = false;
    },

    stop_running: function (record) {
      axios(API.stop_hashid(record.hash_id))
        .then((response) => {
          this.update_states(response.data);
        })
        .catch((error) => {
          console.log(error);
        });
    },

    edit_config: function (record) {
      this.request_visible = true;
      this.current_request_json = Data.parse_config(record);
    },

    set_visibility: function (record, visible) {
      axios(API.set_visibility(record.hash_id, visible))
        .then((response) => {
          this.update_states(response.data);
        })
        .catch((error) => {
          console.log(error);
        });
    },

    on_load_request: function () {
      message.success("Request submitted successfully");
      this.request_visible = false;
      this.get_requests();
    },

    show_pdb: function (record, pdb_path, show_panel) {
      if (pdb_path) {
        if (show_panel) {
          this.pdb_visible = true;
        } else {
          message.info({
            content: "Added to protein viewer, Click to open",
            onClick: () => {
              this.protein_visible = true;
            },
            class: "cursor-pointer",
          });
        }
        this.current_view_pdb = { info: record, pdb_path: pdb_path };
      } else {
        message.warning("No PDB found.");
      }
    },

    pure_name: function (name) {
      return split(name, "___")[0];
    },

    // Actions
    on_select_change: function (selected_hash_ids) {
      this.selected_hash_ids = selected_hash_ids;
    },

    update_states: function (new_requests) {
      forEach(new_requests, (request) => {
        let request_map = this.requests_map[request.hash_id];
        if (request_map) {
          if (request.request_json) {
            assign(request_map, request);
          }
        }
      });
    },

    days_to_ddl: function (record) {
      if (record.request_json.sender.includes("casp")) {
        let name = record.request_json.target;
        if (this.casp_data.by_target && name in this.casp_data.by_target) {
          let target = this.casp_data.by_target[name];
          let ddl = target["Human Exp."];
          let days_to_ddl = moment(ddl).diff(
            moment().subtract(30, "hours"),
            "days"
          );
          return days_to_ddl;
        }
      }
      return -9999;
    },

    from_now_time_abbr: function (time_str) {
      let from_now = moment(time_str).fromNow();
      // from_now = from_now.replace("hours", "h");

      from_now = from_now.replace("minutes", "min");
      from_now = from_now.replace("minute", "min");

      return from_now;
    },

    show_amino_acid_pos: function (ruler_start, ruler_end, L, e) {
      let bound_pos = e.srcElement.getBoundingClientRect();
      let click_pos = e.clientX - bound_pos.x;
      console.log(click_pos);
      let seq_pos = Math.floor(
        ((click_pos - ruler_start) / (ruler_end - ruler_start)) * L
      );
      let text_elem =
        e.srcElement.parentElement.getElementsByClassName("pos_end")[0];
      text_elem.innerHTML = seq_pos;
      text_elem.style = "left: " + click_pos + "px";
    },

    total_time: function (record) {
      // iterate record.state_msgs
      let start_time = record.state_msg[0].time;
      let end_time = record.state_msg[record.state_msg.length - 1].time;
      let duration = moment.duration(end_time - start_time);

      duration = duration.humanize();
      duration = duration.replace("minutes", "min");
      duration = duration.replace("minute", "min");
      duration = duration.replace("a few seconds", "seconds");

      return duration;
    },
  },
};
</script>

<style>
.ant-switch {
  background-color: #d9d9d9 !important;
}

.ant-switch-checked {
  background-color: #1890ff !important;
}
</style>
