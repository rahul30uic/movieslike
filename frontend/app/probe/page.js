import { redirect } from "next/navigation";

// The probe is the home page now.
export default function ProbeRedirect() {
  redirect("/");
}
